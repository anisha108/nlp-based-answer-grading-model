import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class OSDataPreprocessor:
    """
    Preprocessor for Operating Systems question-answer grading dataset
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text == '':
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', str(text)).strip()
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text
    
    def extract_score(self, score_text: str) -> int:
        """Extract numerical score from score text"""
        if pd.isna(score_text):
            return 0
        
        # Look for patterns like "2/5", "2 out of 5", "valued at 2"
        patterns = [
            r'(\d+)/5',
            r'(\d+)\s*out\s*of\s*5',
            r'valued\s*at\s*(\d+)',
            r'score[d]?\s*(\d+)',
            r'^(\d+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(score_text).lower())
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 5)  # Ensure score is between 0-5
        
        return 0
    
    def create_training_format(self, question: str, answer: str, context: str, score: int, feedback: str) -> Dict:
        """Create training format for T5 model"""
        
        # Create input text for T5
        input_text = f"Question: {question}\nStudent Answer: {answer}\nContext: {context}\nTask: Grade and provide feedback"
        
        # Create target text (what model should generate)
        target_text = f"Score: {score}/5\nFeedback: {feedback}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'question': question,
            'student_answer': answer,
            'context': context,
            'score': score,
            'feedback': feedback
        }
    
    def process_dataset(self, csv_path: str) -> pd.DataFrame:
        """Process the main dataset"""
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean the data
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # Extract and clean fields
                question = self.clean_text(row.get('Question', ''))
                user_answer = self.clean_text(row.get('User answer', ''))
                context = self.clean_text(row.get('context', ''))
                
                # Extract score
                score_text = row.get('Answer score(out of 5)', '') or row.get('Output', '')
                score = self.extract_score(score_text)
                
                # Extract feedback
                feedback = self.clean_text(score_text)
                
                # Skip if essential fields are missing
                if not question or not user_answer:
                    continue
                
                # Create training format
                training_sample = self.create_training_format(
                    question, user_answer, context, score, feedback
                )
                
                processed_data.append(training_sample)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        processed_df = pd.DataFrame(processed_data)
        print(f"Processed dataset shape: {processed_df.shape}")
        
        return processed_df
    
    def create_textbook_knowledge_base(self, df: pd.DataFrame) -> List[Dict]:
        """Extract textbook context for knowledge base"""
        knowledge_base = []
        
        for idx, row in df.iterrows():
            context = row.get('context', '')
            if context and len(context.strip()) > 50:  # Only substantial contexts
                
                # Split context into chunks for better retrieval
                if self.nlp:
                    doc = self.nlp(context)
                    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
                else:
                    sentences = sent_tokenize(context)
                
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        knowledge_base.append({
                            'text': sentence.strip(),
                            'source': f'textbook_chunk_{len(knowledge_base)}',
                            'topic': self.extract_topic_from_question(row.get('question', ''))
                        })
        
        return knowledge_base
    
    def extract_topic_from_question(self, question: str) -> str:
        """Extract topic from question for better organization"""
        question_lower = question.lower()
        
        topics = {
            'multithreading': ['thread', 'multithreading', 'concurrent'],
            'memory_management': ['memory', 'paging', 'segmentation', 'virtual'],
            'process_management': ['process', 'scheduling', 'cpu'],
            'file_systems': ['file', 'directory', 'storage'],
            'security': ['security', 'protection', 'kernel mode', 'user mode'],
            'synchronization': ['synchronization', 'deadlock', 'semaphore', 'mutex'],
            'io_systems': ['i/o', 'input', 'output', 'device']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        print(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

def main():
    """Main preprocessing function"""
    preprocessor = OSDataPreprocessor()
    
    # Process both datasets
    datasets = [
        "c:\\Users\\vinay\\Desktop\\research paper\\Operating Systems Paper Evaluation.csv",
        "c:\\Users\\vinay\\Desktop\\research paper\\OS_Evaluation_data.csv"
    ]
    
    all_processed_data = []
    
    for dataset_path in datasets:
        try:
            processed_df = preprocessor.process_dataset(dataset_path)
            all_processed_data.append(processed_df)
        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
    
    # Combine all datasets
    if all_processed_data:
        combined_df = pd.concat(all_processed_data, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['question', 'student_answer'])
        
        print(f"Combined dataset shape: {combined_df.shape}")
        
        # Split dataset
        train_df, val_df, test_df = preprocessor.split_dataset(combined_df)
        
        # Save processed datasets
        train_df.to_csv("c:\\Users\\vinay\\Desktop\\research paper\\train_data.csv", index=False)
        val_df.to_csv("c:\\Users\\vinay\\Desktop\\research paper\\val_data.csv", index=False)
        test_df.to_csv("c:\\Users\\vinay\\Desktop\\research paper\\test_data.csv", index=False)
        
        # Create knowledge base
        knowledge_base = preprocessor.create_textbook_knowledge_base(combined_df)
        
        # Save knowledge base
        kb_df = pd.DataFrame(knowledge_base)
        kb_df.to_csv("c:\\Users\\vinay\\Desktop\\research paper\\knowledge_base.csv", index=False)
        
        print(f"Knowledge base created with {len(knowledge_base)} entries")
        print("Preprocessing completed successfully!")
        
        # Display sample data
        print("\nSample training data:")
        print(train_df[['input_text', 'target_text']].head(2))
        
    else:
        print("No data was successfully processed.")

if __name__ == "__main__":
    main()