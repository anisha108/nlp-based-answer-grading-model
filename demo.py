"""
Demo script to test the OS Grading System with sample data
"""

import pandas as pd
import os
from datetime import datetime

def test_data_loading():
    """Test if the data files can be loaded"""
    print("🔍 Testing data loading...")
    
    files_to_check = [
        "Operating Systems Paper Evaluation.csv",
        "OS_Evaluation_data.csv"
    ]
    
    for file_name in files_to_check:
        file_path = f"c:\\Users\\vinay\\Desktop\\research paper\\{file_name}"
        try:
            df = pd.read_csv(file_path)
            print(f"✅ {file_name}: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)}")
            
            # Show sample data
            if len(df) > 0:
                print(f"   Sample question: {df.iloc[0].get('Question', 'N/A')[:100]}...")
                print(f"   Sample answer: {df.iloc[0].get('User answer', 'N/A')[:100]}...")
            print()
            
        except Exception as e:
            print(f"❌ Error loading {file_name}: {e}")
    
    return True

def run_preprocessing_demo():
    """Run a quick preprocessing demo"""
    print("🔧 Running preprocessing demo...")
    
    try:
        from data_preprocessor import OSDataPreprocessor
        
        preprocessor = OSDataPreprocessor()
        
        # Test with first dataset
        df = preprocessor.process_dataset("c:\\Users\\vinay\\Desktop\\research paper\\Operating Systems Paper Evaluation.csv")
        
        if len(df) > 0:
            print(f"✅ Preprocessed {len(df)} samples")
            print(f"   Sample input: {df.iloc[0]['input_text'][:200]}...")
            print(f"   Sample target: {df.iloc[0]['target_text'][:100]}...")
            
            # Save a small sample for testing
            sample_df = df.head(10)
            sample_df.to_csv("c:\\Users\\vinay\\Desktop\\research paper\\demo_sample.csv", index=False)
            print("   Saved demo sample to demo_sample.csv")
        else:
            print("❌ No data was processed")
            
    except Exception as e:
        print(f"❌ Preprocessing demo failed: {e}")

def create_simple_grader():
    """Create a simple rule-based grader for demonstration"""
    print("🤖 Creating simple demo grader...")
    
    class SimpleGrader:
        def __init__(self):
            self.keywords = {
                'multithreading': ['thread', 'concurrent', 'parallel', 'multiple'],
                'memory': ['memory', 'ram', 'virtual', 'paging', 'segmentation'],
                'process': ['process', 'scheduling', 'cpu', 'execution'],
                'security': ['security', 'protection', 'kernel', 'user mode'],
                'file': ['file', 'directory', 'storage', 'disk']
            }
        
        def grade_answer(self, question, answer):
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            # Determine topic
            topic = 'general'
            for t, keywords in self.keywords.items():
                if any(kw in question_lower for kw in keywords):
                    topic = t
                    break
            
            # Count relevant keywords in answer
            relevant_keywords = self.keywords.get(topic, [])
            keyword_count = sum(1 for kw in relevant_keywords if kw in answer_lower)
            
            # Simple scoring logic
            if len(answer.strip()) < 20:
                score = 0
                feedback = "Answer is too short and lacks detail."
            elif keyword_count == 0:
                score = 1
                feedback = "Answer doesn't contain relevant keywords for this topic."
            elif keyword_count == 1:
                score = 2
                feedback = "Answer shows basic understanding but needs more detail."
            elif keyword_count == 2:
                score = 3
                feedback = "Good answer with relevant concepts mentioned."
            elif keyword_count >= 3:
                score = 4
                feedback = "Excellent answer with comprehensive coverage."
            else:
                score = 2
                feedback = "Answer needs improvement."
            
            return {
                'score': score,
                'feedback': feedback,
                'topic': topic,
                'keywords_found': keyword_count
            }
    
    return SimpleGrader()

def demo_grading():
    """Demonstrate grading with sample questions"""
    print("📝 Demo grading session...")
    
    grader = create_simple_grader()
    
    # Sample questions from your data
    test_cases = [
        {
            'question': "What is multithreading? Explain its benefits.",
            'answer': "Multithreading allows multiple threads to execute concurrently within a single process. Benefits include better resource utilization, improved responsiveness, and ability to perform multiple tasks simultaneously."
        },
        {
            'question': "Explain the difference between kernel mode and user mode.",
            'answer': "Kernel mode has full access to hardware while user mode has restricted access for security."
        },
        {
            'question': "What is virtual memory?",
            'answer': "It's something related to computers."
        }
    ]
    
    print(f"\n{'='*60}")
    print("DEMO GRADING RESULTS")
    print(f"{'='*60}")
    
    for i, test in enumerate(test_cases, 1):
        result = grader.grade_answer(test['question'], test['answer'])
        
        print(f"\n--- Test Case {i} ---")
        print(f"Question: {test['question']}")
        print(f"Answer: {test['answer']}")
        print(f"Score: {result['score']}/5")
        print(f"Topic: {result['topic']}")
        print(f"Keywords Found: {result['keywords_found']}")
        print(f"Feedback: {result['feedback']}")

def main():
    """Main demo function"""
    print("🎓 OS Answer Grading System - Demo")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    # Test 1: Data loading
    test_data_loading()
    
    # Test 2: Preprocessing demo
    run_preprocessing_demo()
    
    # Test 3: Simple grading demo
    demo_grading()
    
    print(f"\n{'='*60}")
    print("✅ Demo completed successfully!")
    print(f"{'='*60}")
    
    print("\n📋 Next Steps:")
    print("1. Run 'python run_pipeline.py' for complete setup")
    print("2. Or run individual components:")
    print("   - python data_preprocessor.py")
    print("   - python rag_system.py") 
    print("   - python model_trainer.py")
    print("   - python grading_system.py")
    
    print(f"\nDemo completed at: {datetime.now()}")

if __name__ == "__main__":
    main()