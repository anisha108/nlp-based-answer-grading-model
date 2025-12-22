"""
Test script for the Automatic Short Answer Grading System API
Run this after starting the FastAPI server to verify endpoints work correctly.
"""

import requests
import json
import os
import sys

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"  Health Status: {data.get('status')}")
        print(f"  Knowledge Base Loaded: {data.get('knowledge_base_loaded')}")
        print(f"  OpenAI Configured: {data.get('openai_configured')}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_upload_pdf(pdf_path):
    """Test uploading a PDF file"""
    print("\n" + "="*60)
    print("Testing PDF Upload Endpoint")
    print("="*60)
    
    if not os.path.exists(pdf_path):
        print(f"✗ Error: PDF file not found at {pdf_path}")
        print("  Please provide a valid PDF file path.")
        return False
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            response.raise_for_status()
        
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"  Message: {data.get('message')}")
        print(f"  File: {data.get('file_name')}")
        print(f"  Pages Processed: {data.get('total_pages')}")
        print(f"  Chunks Created: {data.get('chunks_processed')}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_grading(question, student_answer):
    """Test the grading endpoint"""
    print("\n" + "="*60)
    print("Testing Grading Endpoint")
    print("="*60)
    
    payload = {
        "question": question,
        "student_answer": student_answer
    }
    
    print(f"\nQuestion: {question}")
    print(f"Student Answer: {student_answer}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/grade",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"\n✓ Status: {response.status_code}")
        print(f"\n📊 GRADING RESULT:")
        print(f"  Score: {data.get('score')}/5")
        print(f"\n  Feedback:")
        print(f"  {data.get('feedback')}")
        
        return True
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"  Detail: {error_detail.get('detail')}")
            except:
                print(f"  Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def run_sample_tests():
    """Run a series of sample tests"""
    print("\n" + "="*70)
    print(" AUTOMATIC SHORT ANSWER GRADING SYSTEM - API TEST SUITE")
    print("="*70)
    
    # Test 1: Health Check
    if not test_health_check():
        print("\n⚠ Server might not be running. Start it with: python main.py")
        return
    
    # Test 2: Upload (optional - requires user to provide PDF)
    print("\n" + "-"*60)
    print("Note: To test upload, provide a PDF file path as argument:")
    print("  python test_api.py <path_to_pdf>")
    print("-"*60)
    
    # Test 3: Sample grading questions
    print("\n" + "="*60)
    print("Running Sample Grading Tests")
    print("="*60)
    
    test_cases = [
        {
            "question": "What is a process in an operating system?",
            "answer": "A process is a program in execution with its own memory space."
        },
        {
            "question": "Explain the difference between a thread and a process.",
            "answer": "Threads share the same memory space while processes have separate memory spaces. Threads are lighter weight than processes."
        },
        {
            "question": "What is deadlock?",
            "answer": "When two processes wait for each other forever."
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} of {len(test_cases)} ---")
        success = test_grading(test_case["question"], test_case["answer"])
        results.append(success)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    # Check if PDF path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # Run health check first
        test_health_check()
        
        # Test upload with provided PDF
        test_upload_pdf(pdf_path)
        
        # Ask if user wants to run grading tests
        try:
            choice = input("\nRun sample grading tests? (y/n): ")
            if choice.lower() == 'y':
                test_cases = [
                    {
                        "question": "What is a process in an operating system?",
                        "answer": "A process is a program in execution with its own memory space."
                    }
                ]
                for test_case in test_cases:
                    test_grading(test_case["question"], test_case["answer"])
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user.")
    else:
        # Run standard test suite
        run_sample_tests()
