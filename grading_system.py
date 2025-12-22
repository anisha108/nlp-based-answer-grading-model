import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import re
from typing import Dict, List, Optional
import os
from rag_system import RAGKnowledgeBase
import json
from datetime import datetime

class OSGradingSystem:
    """
    Complete Operating Systems Answer Grading System
    """
    
    def __init__(self, model_path: str, rag_kb_path: str = None):
        """
        Initialize the grading system
        
        Args:
            model_path: Path to the trained T5 model
            rag_kb_path: Path to the RAG knowledge base
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize RAG system
        self.rag_system = None
        if rag_kb_path and os.path.exists(f"{rag_kb_path}_data.pkl"):
            try:
                self.rag_system = RAGKnowledgeBase()
                self.rag_system.load_knowledge_base(rag_kb_path)
                print("RAG knowledge base loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load RAG system: {e}")
        
        # Grading rubric for different score levels
        self.rubric = {
            0: "Completely incorrect or irrelevant answer",
            1: "Mostly incorrect with minimal understanding shown",
            2: "Partially correct but missing key concepts or has significant errors",
            3: "Generally correct but lacks detail or has minor errors",
            4: "Mostly correct and well-explained with minor gaps",
            5: "Excellent answer that is complete, accurate, and well-explained"
        }
    
    def extract_score_and_feedback(self, prediction: str) -> Dict:
        """
        Extract score and feedback from model prediction
        """
        # Extract score
        score_patterns = [
            r'Score:\s*(\d+)/5',
            r'Score:\s*(\d+)',
            r'(\d+)/5',
            r'scored?\s*(\d+)',
            r'grade[d]?\s*(\d+)'
        ]
        
        score = 0
        for pattern in score_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                score = min(max(score, 0), 5)  # Ensure score is between 0-5
                break
        
        # Extract feedback
        feedback_patterns = [
            r'Feedback:\s*(.+?)(?:\n|$)',
            r'feedback[:\s]+(.+?)(?:\n|$)',
            r'(?:Score:\s*\d+/5\s*)(.+)',
            r'(?:Score:\s*\d+\s*)(.+)'
        ]
        
        feedback = ""
        for pattern in feedback_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                feedback = match.group(1).strip()
                break
        
        if not feedback:
            feedback = prediction.strip()
        
        return {
            'score': score,
            'feedback': feedback,
            'raw_prediction': prediction
        }
    
    def enhance_feedback(self, score: int, feedback: str, question: str, student_answer: str) -> str:
        """
        Enhance feedback with additional suggestions and rubric information
        """
        enhanced_feedback = feedback
        
        # Add rubric explanation
        rubric_explanation = self.rubric.get(score, "")
        if rubric_explanation:
            enhanced_feedback += f"\n\nRubric Level {score}: {rubric_explanation}"
        
        # Add specific suggestions based on score
        if score <= 2:
            enhanced_feedback += "\n\nSuggestions for improvement:"
            enhanced_feedback += "\n- Review the fundamental concepts related to this topic"
            enhanced_feedback += "\n- Make sure your answer directly addresses the question asked"
            enhanced_feedback += "\n- Include key terminology and definitions"
            
        elif score == 3:
            enhanced_feedback += "\n\nTo improve your answer:"
            enhanced_feedback += "\n- Add more specific details and examples"
            enhanced_feedback += "\n- Ensure all parts of the question are addressed"
            enhanced_feedback += "\n- Check for any technical inaccuracies"
            
        elif score == 4:
            enhanced_feedback += "\n\nGood work! To reach excellence:"
            enhanced_feedback += "\n- Consider adding more depth to your explanation"
            enhanced_feedback += "\n- Include relevant examples or use cases"
            enhanced_feedback += "\n- Ensure complete coverage of all aspects"
        
        return enhanced_feedback
    
    def grade_answer(self, question: str, student_answer: str, 
                    include_context: bool = True, max_context_length: int = 800) -> Dict:
        """
        Grade a single student answer
        
        Args:
            question: The question being asked
            student_answer: Student's answer to grade
            include_context: Whether to include RAG context
            max_context_length: Maximum length of context to include
            
        Returns:
            Dictionary with grading results
        """
        
        # Get relevant context if RAG system is available
        context = ""
        if include_context and self.rag_system:
            context = self.rag_system.get_context_for_question(question, max_context_length)
        
        # Format input for the model
        input_text = f"Question: {question}\nStudent Answer: {student_answer}\nContext: {context}\nTask: Grade and provide feedback"
        
        # Tokenize input
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
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode prediction
        prediction = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract score and feedback
        result = self.extract_score_and_feedback(prediction)
        
        # Enhance feedback
        enhanced_feedback = self.enhance_feedback(
            result['score'], result['feedback'], question, student_answer
        )
        
        return {
            'question': question,
            'student_answer': student_answer,
            'score': result['score'],
            'max_score': 5,
            'feedback': enhanced_feedback,
            'context_used': context,
            'raw_prediction': result['raw_prediction'],
            'timestamp': datetime.now().isoformat()
        }
    
    def grade_batch(self, questions_answers: List[Dict], 
                   include_context: bool = True) -> List[Dict]:
        """
        Grade a batch of question-answer pairs
        
        Args:
            questions_answers: List of dicts with 'question' and 'student_answer' keys
            include_context: Whether to include RAG context
            
        Returns:
            List of grading results
        """
        results = []
        
        print(f"Grading {len(questions_answers)} answers...")
        
        for i, qa in enumerate(questions_answers):
            print(f"Grading answer {i+1}/{len(questions_answers)}")
            
            result = self.grade_answer(
                question=qa['question'],
                student_answer=qa['student_answer'],
                include_context=include_context
            )
            
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict], save_path: str = None) -> Dict:
        """
        Generate a comprehensive grading report
        
        Args:
            results: List of grading results
            save_path: Optional path to save the report
            
        Returns:
            Report dictionary
        """
        if not results:
            return {}
        
        scores = [r['score'] for r in results]
        
        report = {
            'summary': {
                'total_questions': len(results),
                'average_score': sum(scores) / len(scores),
                'median_score': sorted(scores)[len(scores)//2],
                'score_distribution': {str(i): scores.count(i) for i in range(6)},
                'pass_rate': sum(1 for s in scores if s >= 3) / len(scores) * 100
            },
            'detailed_results': results,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_path}")
        
        return report
    
    def interactive_grading(self):
        """
        Interactive grading session
        """
        print("=== OS Answer Grading System ===")
        print("Enter 'quit' to exit")
        
        while True:
            print("\n" + "="*50)
            question = input("Enter the question: ").strip()
            
            if question.lower() == 'quit':
                break
            
            student_answer = input("Enter the student's answer: ").strip()
            
            if student_answer.lower() == 'quit':
                break
            
            print("\nGrading...")
            result = self.grade_answer(question, student_answer)
            
            print(f"\n{'='*50}")
            print(f"GRADING RESULT")
            print(f"{'='*50}")
            print(f"Score: {result['score']}/5")
            print(f"\nFeedback:")
            print(result['feedback'])
            print(f"{'='*50}")

def main():
    """Main function for testing the grading system"""
    
    # Initialize grading system
    model_path = "c:\\Users\\vinay\\Desktop\\research paper\\models\\best_model"
    rag_kb_path = "c:\\Users\\vinay\\Desktop\\research paper\\rag_kb"
    
    try:
        grading_system = OSGradingSystem(model_path, rag_kb_path)
    except Exception as e:
        print(f"Error initializing grading system: {e}")
        print("Make sure you have trained the model first by running model_trainer.py")
        return
    
    # Test with sample questions
    test_questions = [
        {
            'question': "What is multithreading and explain its benefits?",
            'student_answer': "Multithreading allows a program to execute multiple threads concurrently. Benefits include better resource utilization, improved responsiveness, and parallel execution on multicore systems."
        },
        {
            'question': "Explain the difference between kernel mode and user mode.",
            'student_answer': "Kernel mode has full access to hardware and system resources, while user mode has restricted access for security. Programs switch between modes using system calls."
        },
        {
            'question': "What is virtual memory?",
            'student_answer': "Virtual memory is a technique that uses disk space as an extension of RAM, allowing programs to use more memory than physically available."
        }
    ]
    
    # Grade the test questions
    results = grading_system.grade_batch(test_questions)
    
    # Display results
    for result in results:
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"Student Answer: {result['student_answer']}")
        print(f"Score: {result['score']}/5")
        print(f"Feedback: {result['feedback']}")
    
    # Generate and save report
    report = grading_system.generate_report(
        results, 
        "c:\\Users\\vinay\\Desktop\\research paper\\grading_report.json"
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total Questions: {report['summary']['total_questions']}")
    print(f"Average Score: {report['summary']['average_score']:.2f}/5")
    print(f"Pass Rate: {report['summary']['pass_rate']:.1f}%")
    print(f"Score Distribution: {report['summary']['score_distribution']}")
    
    # Start interactive session
    choice = input("\nWould you like to start an interactive grading session? (y/n): ")
    if choice.lower() == 'y':
        grading_system.interactive_grading()

if __name__ == "__main__":
    main()