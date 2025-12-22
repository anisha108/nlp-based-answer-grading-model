"""
Test script to verify the OS Grading System components
"""

import pandas as pd
import os
from datetime import datetime

def test_data_files():
    """Test if all data files were created correctly"""
    print("🔍 Testing data files...")
    
    files_to_check = [
        ("train_data.csv", "Training data"),
        ("val_data.csv", "Validation data"),
        ("test_data.csv", "Test data"),
        ("knowledge_base.csv", "Knowledge base"),
        ("enhanced_train_data.csv", "Enhanced training data"),
        ("rag_kb_data.pkl", "RAG knowledge base"),
        ("rag_kb_faiss.index", "FAISS index")
    ]
    
    all_good = True
    
    for filename, description in files_to_check:
        filepath = f"c:\\Users\\vinay\\Desktop\\research paper\\{filename}"
        if os.path.exists(filepath):
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath)
                    print(f"✅ {description}: {len(df)} rows")
                except Exception as e:
                    print(f"❌ {description}: File exists but can't read - {e}")
                    all_good = False
            else:
                file_size = os.path.getsize(filepath) / (1024*1024)  # MB
                print(f"✅ {description}: {file_size:.1f} MB")
        else:
            print(f"❌ {description}: Not found")
            all_good = False
    
    return all_good

def test_rag_system():
    """Test the RAG system"""
    print("\n📚 Testing RAG system...")
    
    try:
        from rag_system import RAGKnowledgeBase
        
        # Initialize RAG system
        rag = RAGKnowledgeBase()
        
        # Load knowledge base
        rag.load_knowledge_base("c:\\Users\\vinay\\Desktop\\research paper\\rag_kb")
        
        if rag.is_built:
            print("✅ RAG system loaded successfully")
            
            # Test retrieval
            test_questions = [
                "What is multithreading?",
                "How does memory management work?",
                "Explain process scheduling"
            ]
            
            for question in test_questions:
                context = rag.get_context_for_question(question, max_context_length=200)
                if context:
                    print(f"✅ Retrieved context for: {question[:30]}...")
                else:
                    print(f"⚠️ No context found for: {question[:30]}...")
            
            return True
        else:
            print("❌ RAG system failed to load")
            return False
            
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False

def test_simple_grading():
    """Test simple grading functionality"""
    print("\n🤖 Testing simple grading...")
    
    try:
        # Create a simple grader (like in demo)
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
        
        grader = SimpleGrader()
        
        # Test with sample questions
        test_cases = [
            {
                'question': "What is multithreading and explain its benefits?",
                'answer': "Multithreading allows multiple threads to execute concurrently within a single process. Benefits include better resource utilization, improved responsiveness, and ability to perform multiple tasks simultaneously."
            },
            {
                'question': "Explain virtual memory management.",
                'answer': "Virtual memory is a memory management technique that uses disk space as an extension of RAM."
            }
        ]
        
        all_passed = True
        for i, test in enumerate(test_cases, 1):
            result = grader.grade_answer(test['question'], test['answer'])
            if result['score'] > 0:
                print(f"✅ Test case {i}: Score {result['score']}/5")
            else:
                print(f"❌ Test case {i}: Failed to grade properly")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Simple grading error: {e}")
        return False

def test_data_quality():
    """Test the quality of processed data"""
    print("\n📊 Testing data quality...")
    
    try:
        # Load training data
        train_df = pd.read_csv("c:\\Users\\vinay\\Desktop\\research paper\\train_data.csv")
        
        print(f"✅ Training data loaded: {len(train_df)} samples")
        
        # Check for required columns
        required_cols = ['input_text', 'target_text', 'question', 'student_answer', 'score']
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print("✅ All required columns present")
        
        # Check data quality
        empty_inputs = train_df['input_text'].isna().sum()
        empty_targets = train_df['target_text'].isna().sum()
        
        if empty_inputs > 0:
            print(f"⚠️ {empty_inputs} empty input texts")
        else:
            print("✅ No empty input texts")
        
        if empty_targets > 0:
            print(f"⚠️ {empty_targets} empty target texts")
        else:
            print("✅ No empty target texts")
        
        # Check score distribution
        score_dist = train_df['score'].value_counts().sort_index()
        print(f"✅ Score distribution: {dict(score_dist)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 OS Answer Grading System - Component Tests")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    tests = [
        ("Data Files", test_data_files),
        ("RAG System", test_rag_system),
        ("Simple Grading", test_simple_grading),
        ("Data Quality", test_data_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\n📋 Next steps:")
        print("1. Wait for model training to complete")
        print("2. Run: python grading_system.py")
        print("3. Start using the grading system!")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()