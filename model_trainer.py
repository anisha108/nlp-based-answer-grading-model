import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
from typing import Dict, List, Tuple
import os
import json
from tqdm import tqdm
import wandb
from datetime import datetime

class OSGradingDataset(Dataset):
    """
    Dataset class for OS question grading
    """
    
    def __init__(self, data_df: pd.DataFrame, tokenizer: T5Tokenizer, max_input_length: int = 512, max_target_length: int = 256):
        self.data = data_df
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        input_text = row['input_text']
        target_text = row['target_text']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class OSGradingTrainer:
    """
    Trainer class for OS question grading model
    """
    
    def __init__(self, model_name: str = 't5-base', device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Add special tokens if needed
        special_tokens = ['<question>', '<answer>', '<context>', '<score>', '<feedback>']
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare datasets for training
        """
        train_dataset = OSGradingDataset(train_df, self.tokenizer)
        val_dataset = OSGradingDataset(val_df, self.tokenizer)
        test_dataset = OSGradingDataset(test_df, self.tokenizer)
        
        return train_dataset, val_dataset, test_dataset
    
    def extract_score_from_prediction(self, prediction: str) -> int:
        """
        Extract numerical score from model prediction
        """
        # Look for score patterns
        patterns = [
            r'Score:\s*(\d+)/5',
            r'Score:\s*(\d+)',
            r'(\d+)/5',
            r'scored?\s*(\d+)',
            r'grade[d]?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 5)  # Ensure score is between 0-5
        
        return 0  # Default score if no pattern found
    
    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict:
        """
        Compute evaluation metrics
        """
        pred_scores = [self.extract_score_from_prediction(pred) for pred in predictions]
        true_scores = [self.extract_score_from_prediction(target) for target in targets]
        
        # Compute metrics
        mse = mean_squared_error(true_scores, pred_scores)
        mae = mean_absolute_error(true_scores, pred_scores)
        
        # Compute accuracy (exact match)
        exact_match = sum(1 for p, t in zip(pred_scores, true_scores) if p == t) / len(pred_scores)
        
        # Compute accuracy within 1 point
        within_1 = sum(1 for p, t in zip(pred_scores, true_scores) if abs(p - t) <= 1) / len(pred_scores)
        
        return {
            'mse': mse,
            'mae': mae,
            'exact_match_accuracy': exact_match,
            'within_1_accuracy': within_1,
            'avg_pred_score': np.mean(pred_scores),
            'avg_true_score': np.mean(true_scores)
        }
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 5e-5,
              save_dir: str = "c:\\Users\\vinay\\Desktop\\research paper\\models"):
        """
        Train the model
        """
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize wandb for experiment tracking
        wandb.init(
            project="os-grading-system",
            config={
                "model_name": self.model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset)
            }
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch in train_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation")
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()
                    
                    # Generate predictions for evaluation
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=256,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode predictions and targets
                    batch_predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    batch_targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    val_predictions.extend(batch_predictions)
                    val_targets.extend(batch_targets)
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Compute metrics
            metrics = self.compute_metrics(val_predictions, val_targets)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
            print(f"Within 1 Point Accuracy: {metrics['within_1_accuracy']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **metrics
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(save_dir, "best_model")
                self.model.save_pretrained(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)
                print(f"Best model saved to {model_save_path}")
        
        wandb.finish()
        print("Training completed!")
    
    def evaluate(self, test_dataset: Dataset, model_path: str = None) -> Dict:
        """
        Evaluate the model on test set
        """
        if model_path:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
        
        self.model.eval()
        
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Generate predictions
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode predictions and targets
                batch_predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                predictions.extend(batch_predictions)
                targets.extend(batch_targets)
        
        # Compute final metrics
        metrics = self.compute_metrics(predictions, targets)
        
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, predictions, targets
    
    def predict(self, question: str, student_answer: str, context: str = "") -> Dict:
        """
        Make a prediction for a single question-answer pair
        """
        self.model.eval()
        
        # Format input
        input_text = f"Question: {question}\nStudent Answer: {student_answer}\nContext: {context}\nTask: Grade and provide feedback"
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode prediction
        prediction = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract score and feedback
        score = self.extract_score_from_prediction(prediction)
        
        return {
            'prediction': prediction,
            'score': score,
            'input_text': input_text
        }

def main():
    """Main training function"""
    
    # Load datasets
    try:
        train_df = pd.read_csv("c:\\Users\\vinay\\Desktop\\research paper\\enhanced_train_data.csv")
        val_df = pd.read_csv("c:\\Users\\vinay\\Desktop\\research paper\\val_data.csv")
        test_df = pd.read_csv("c:\\Users\\vinay\\Desktop\\research paper\\test_data.csv")
        
        print(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Please run data_preprocessor.py and rag_system.py first.")
        return
    
    # Initialize trainer
    trainer = OSGradingTrainer(model_name='t5-base')
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(train_df, val_df, test_df)
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=5,
        batch_size=4,  # Adjust based on GPU memory
        learning_rate=3e-4
    )
    
    # Evaluate on test set
    best_model_path = "c:\\Users\\vinay\\Desktop\\research paper\\models\\best_model"
    metrics, predictions, targets = trainer.evaluate(test_dataset, best_model_path)
    
    # Save evaluation results
    results = {
        'metrics': metrics,
        'predictions': predictions[:10],  # Save first 10 for inspection
        'targets': targets[:10]
    }
    
    with open("c:\\Users\\vinay\\Desktop\\research paper\\evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()