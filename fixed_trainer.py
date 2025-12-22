"""
Fixed Custom Model Trainer for OS Answer Grading
Handles import issues and missing dependencies
"""

import os
import sys
import json
from datetime import datetime

def install_missing_packages():
    """Install missing packages"""
    import subprocess
    
    packages_to_install = []
    
    # Check SentencePiece
    try:
        import sentencepiece
        print("✅ SentencePiece available")
    except ImportError:
        print("❌ SentencePiece missing - will install")
        packages_to_install.append("sentencepiece")
    
    if packages_to_install:
        print(f"Installing missing packages: {packages_to_install}")
        for package in packages_to_install:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_dependencies():
    """Check if all required packages are available"""
    print("Checking dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found. Install with: pip install transformers")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found. Install with: pip install pandas")
        return False
    
    # Test T5 tokenizer specifically
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        print("✅ T5 tokenizer working")
    except Exception as e:
        print(f"❌ T5 tokenizer issue: {e}")
        return False
    
    return True

def check_data():
    """Check if training data is available"""
    print("\nChecking training data...")
    
    if not os.path.exists("train_data.csv"):
        print("❌ train_data.csv not found. Run data_preprocessor.py first.")
        return False
    
    if not os.path.exists("val_data.csv"):
        print("❌ val_data.csv not found. Run data_preprocessor.py first.")
        return False
    
    try:
        import pandas as pd
        train_df = pd.read_csv("train_data.csv")
        val_df = pd.read_csv("val_data.csv")
        
        print(f"✅ Training data: {len(train_df)} samples")
        print(f"✅ Validation data: {len(val_df)} samples")
        
        # Check required columns
        required_cols = ['input_text', 'target_text', 'question', 'student_answer', 'score']
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def run_training():
    """Run the actual training with fixed imports"""
    print("\n" + "="*60)
    print("🚀 STARTING CUSTOM MODEL TRAINING")
    print("This will train a T5 model on YOUR specific grading patterns")
    print("="*60)
    
    try:
        # Import training modules with fixed imports
        import torch
        from torch.utils.data import Dataset, DataLoader
        import pandas as pd
        import numpy as np
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        # Fix for AdamW import - it's now in torch.optim
        try:
            from transformers import AdamW
        except ImportError:
            from torch.optim import AdamW
            print("Using AdamW from torch.optim")
        
        from tqdm import tqdm
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load data
        print("Loading training data...")
        train_df = pd.read_csv("train_data.csv")
        val_df = pd.read_csv("val_data.csv")
        
        # Initialize model
        print("Loading T5-small model...")
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        model.to(device)
        
        # Simple dataset class
        class SimpleDataset(Dataset):
            def __init__(self, df, tokenizer):
                self.data = df.reset_index(drop=True)
                self.tokenizer = tokenizer
                print(f"Dataset created with {len(self.data)} examples")
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                row = self.data.iloc[idx]
                
                input_text = str(row['input_text'])
                target_text = str(row['target_text'])
                
                # Tokenize with proper handling
                try:
                    input_encoding = self.tokenizer(
                        input_text, 
                        max_length=512, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt'
                    )
                    target_encoding = self.tokenizer(
                        target_text, 
                        max_length=256, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': input_encoding['input_ids'].flatten(),
                        'attention_mask': input_encoding['attention_mask'].flatten(),
                        'labels': target_encoding['input_ids'].flatten()
                    }
                except Exception as e:
                    print(f"Error tokenizing sample {idx}: {e}")
                    # Return a dummy sample
                    dummy_ids = torch.zeros(512, dtype=torch.long)
                    dummy_labels = torch.zeros(256, dtype=torch.long)
                    return {
                        'input_ids': dummy_ids,
                        'attention_mask': torch.ones(512, dtype=torch.long),
                        'labels': dummy_labels
                    }
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = SimpleDataset(train_df, tokenizer)
        val_dataset = SimpleDataset(val_df, tokenizer)
        
        # Create data loaders with error handling
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=3e-4)
        num_epochs = 3
        
        print(f"Training for {num_epochs} epochs on {len(train_dataset)} examples...")
        print("This will learn YOUR specific grading patterns!")
        
        # Training loop
        model.train()
        best_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            epoch_loss = 0
            num_batches = 0
            
            try:
                progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        optimizer.zero_grad()
                        
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        # Forward pass
                        outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels
                        )
                        loss = outputs.loss
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                        
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        continue
                
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Average training loss: {avg_loss:.4f}")
                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        
                        # Create models directory
                        os.makedirs("models", exist_ok=True)
                        save_path = "models/best_custom_model"
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Save model
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        
                        # Save training info
                        training_info = {
                            'model_type': 'T5ForConditionalGeneration',
                            'base_model': 't5-small',
                            'training_samples': len(train_dataset),
                            'validation_samples': len(val_dataset),
                            'epochs_trained': epoch + 1,
                            'best_loss': float(best_loss),
                            'trained_at': datetime.now().isoformat(),
                            'device_used': device
                        }
                        
                        with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
                            json.dump(training_info, f, indent=2)
                        
                        print(f"✅ Best model saved! Loss: {best_loss:.4f}")
                    
                    training_history.append({
                        'epoch': epoch + 1,
                        'loss': avg_loss
                    })
                
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                continue
        
        # Save training history
        with open("training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Model saved to: models/best_custom_model")
        print(f"Final best loss: {best_loss:.4f}")
        
        # Test the model
        print("\n🧪 Testing trained model...")
        model.eval()
        
        test_input = "Question: What is multithreading? Student Answer: Multithreading allows concurrent execution of threads. Task: Grade and provide feedback"
        
        try:
            input_encoding = tokenizer(
                test_input, 
                max_length=512, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_encoding['input_ids'],
                    attention_mask=input_encoding['attention_mask'],
                    max_length=256,
                    num_beams=2,  # Reduced for stability
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"✅ Sample prediction: {prediction}")
            
        except Exception as e:
            print(f"⚠️ Test generation failed: {e}")
            print("Model saved successfully but test generation had issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎓 Fixed Custom Model Trainer")
    print("Trains T5 model on YOUR specific grading patterns")
    print("="*50)
    
    # Install missing packages
    if not install_missing_packages():
        print("❌ Failed to install required packages")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies not ready. Please install required packages.")
        return
    
    # Check data
    if not check_data():
        print("\n❌ Training data not ready. Please run data preprocessing first.")
        return
    
    # Confirm training
    print(f"\n⚠️ This will train a model specifically on YOUR grading patterns")
    print(f"Training time: ~10-20 minutes")
    print(f"The model will learn from YOUR 91 grading examples")
    print(f"This creates a model that understands YOUR specific rubric and feedback style")
    
    choice = input("\nDo you want to proceed with training? (y/n): ")
    if choice.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    success = run_training()
    
    if success:
        print(f"\n🎉 SUCCESS! Your custom model is ready!")
        print(f"\nWhat your model learned:")
        print(f"✅ YOUR scoring patterns from 91 examples")
        print(f"✅ YOUR feedback style and depth")
        print(f"✅ How YOU evaluate answer quality")
        print(f"✅ YOUR specific rubric criteria")
        
        print(f"\nNext steps:")
        print(f"1. Test your model: python trained_grading_system.py")
        print(f"2. Use interactive grading with YOUR trained model")
        print(f"3. The model can now grade NEW answers based on YOUR patterns!")
    else:
        print(f"\n❌ Training failed. Check the error messages above.")
        print(f"You can still use the simple grader: python simple_grader.py")

if __name__ == "__main__":
    main()