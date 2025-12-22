"""
Simple OS Answer Grading System
Works without the trained T5 model using rule-based grading + RAG context
"""

import pandas as pd
import re
from typing import Dict, List
from datetime import datetime
import json
import os

try:
    from rag_system import RAGKnowledgeBase
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG system not available")

class SimpleOSGrader:
    """
    Simple rule-based grading system for OS questions
    """
    
    def __init__(self, use_rag: bool = True):
        """Initialize the simple grader"""
        
        # OS topic keywords and concepts
        self.topic_keywords = {
            'multithreading': {
                'keywords': ['thread', 'multithreading', 'concurrent', 'parallel', 'synchronization'],
                'concepts': ['concurrency', 'parallelism', 'thread safety', 'race condition', 'deadlock']
            },
            'memory_management': {
                'keywords': ['memory', 'ram', 'virtual', 'paging', 'segmentation', 'allocation'],
                'concepts': ['virtual memory', 'page fault', 'memory leak', 'fragmentation', 'swapping']
            },
            'process_management': {
                'keywords': ['process', 'scheduling', 'cpu', 'execution', 'context', 'switch'],
                'concepts': ['process state', 'scheduler', 'preemption', 'time slice', 'priority']
            },
            'file_systems': {
                'keywords': ['file', 'directory', 'storage', 'disk', 'filesystem', 'inode'],
                'concepts': ['file allocation', 'directory structure', 'metadata', 'journaling']
            },
            'security': {
                'keywords': ['security', 'protection', 'kernel', 'user', 'mode', 'privilege'],
                'concepts': ['access control', 'authentication', 'authorization', 'privilege escalation']
            },
            'synchronization': {
                'keywords': ['synchronization', 'mutex', 'semaphore', 'lock', 'critical', 'section'],
                'concepts': ['mutual exclusion', 'deadlock prevention', 'starvation', 'priority inversion']
            },
            'io_systems': {
                'keywords': ['input', 'output', 'device', 'driver', 'interrupt', 'dma'],
                'concepts': ['device controller', 'interrupt handling', 'buffering', 'spooling']
            }
        }
        
        # Grading rubric
        self.rubric = {
            0: "Completely incorrect or irrelevant answer",
            1: "Mostly incorrect with minimal understanding shown",
            2: "Partially correct but missing key concepts or has significant errors",
            3: "Generally correct but lacks detail or has minor errors",
            4: "Mostly correct and well-explained with minor gaps",
            5: "Excellent answer that is complete, accurate, and well-explained"
        }
        
        # Initialize RAG system if available
        self.rag_system = None
        if use_rag and RAG_AVAILABLE:
            try:
                self.rag_system = RAGKnowledgeBase()
                rag_path = "c:\\Users\\vinay\\Desktop\\research paper\\rag_kb"
                if os.path.exists(f"{rag_path}_data.pkl"):
                    self.rag_system.load_knowledge_base(rag_path)
                    print("✅ RAG system loaded for enhanced grading")
                else:
                    self.rag_system = None
                    print("⚠️ RAG knowledge base not found, using basic grading")
            except Exception as e:
                print(f"⚠️ Could not load RAG system: {e}")
                self.rag_system = None
    
    def identify_topic(self, question: str) -> str:
        """Identify the main topic of the question"""
        question_lower = question.lower()
        
        topic_scores = {}
        for topic, data in self.topic_keywords.items():
            score = 0
            # Check keywords
            for keyword in data['keywords']:
                if keyword in question_lower:
                    score += 2
            # Check concepts
            for concept in data['concepts']:
                if concept in question_lower:
                    score += 1
            
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        else:
            return 'general'
    
    def analyze_answer_content(self, answer: str, topic: str) -> Dict:
        """Analyze the content quality of the answer"""
        answer_lower = answer.lower()
        
        # Basic checks
        word_count = len(answer.split())
        if word_count < 10:
            return {
                'content_score': 0,
                'issues': ['Answer too short (less than 10 words)'],
                'strengths': []
            }
        
        # Topic-specific analysis
        topic_data = self.topic_keywords.get(topic, {'keywords': [], 'concepts': []})
        
        keyword_matches = []
        concept_matches = []
        
        for keyword in topic_data['keywords']:
            if keyword in answer_lower:
                keyword_matches.append(keyword)
        
        for concept in topic_data['concepts']:
            if concept in answer_lower:
                concept_matches.append(concept)
        
        # Calculate content score
        keyword_score = min(len(keyword_matches) * 0.5, 2.0)  # Max 2 points for keywords
        concept_score = min(len(concept_matches) * 0.5, 2.0)  # Max 2 points for concepts
        length_score = min(word_count / 50.0, 1.0)  # Max 1 point for length
        
        content_score = keyword_score + concept_score + length_score
        
        # Identify strengths and issues
        strengths = []
        issues = []
        
        if keyword_matches:
            strengths.append(f"Uses relevant terminology: {', '.join(keyword_matches[:3])}")
        else:
            issues.append("Missing key terminology for this topic")
        
        if concept_matches:
            strengths.append(f"Mentions important concepts: {', '.join(concept_matches[:3])}")
        
        if word_count >= 50:
            strengths.append("Provides detailed explanation")
        elif word_count < 20:
            issues.append("Answer lacks sufficient detail")
        
        return {
            'content_score': min(content_score, 5.0),
            'keyword_matches': keyword_matches,
            'concept_matches': concept_matches,
            'word_count': word_count,
            'strengths': strengths,
            'issues': issues
        }
    
    def get_rag_context(self, question: str) -> str:
        """Get relevant context from RAG system"""
        if self.rag_system and self.rag_system.is_built:
            return self.rag_system.get_context_for_question(question, max_context_length=300)
        return ""
    
    def generate_feedback(self, question: str, answer: str, score: int, analysis: Dict, topic: str) -> str:
        """Generate detailed feedback for the student"""
        
        feedback_parts = []
        
        # Score explanation
        feedback_parts.append(f"Score: {score}/5 - {self.rubric[score]}")
        
        # Strengths
        if analysis['strengths']:
            feedback_parts.append("\n✅ Strengths:")
            for strength in analysis['strengths']:
                feedback_parts.append(f"  • {strength}")
        
        # Issues to address
        if analysis['issues']:
            feedback_parts.append("\n⚠️ Areas for improvement:")
            for issue in analysis['issues']:
                feedback_parts.append(f"  • {issue}")
        
        # Specific suggestions based on score
        feedback_parts.append("\n💡 Suggestions:")
        
        if score <= 2:
            feedback_parts.append("  • Review the fundamental concepts of this topic")
            feedback_parts.append("  • Make sure your answer directly addresses the question")
            feedback_parts.append("  • Include key terminology and definitions")
        elif score == 3:
            feedback_parts.append("  • Add more specific details and examples")
            feedback_parts.append("  • Ensure all parts of the question are addressed")
            feedback_parts.append("  • Check for any technical inaccuracies")
        elif score == 4:
            feedback_parts.append("  • Consider adding more depth to your explanation")
            feedback_parts.append("  • Include relevant examples or use cases")
            feedback_parts.append("  • Ensure complete coverage of all aspects")
        
        # Topic-specific suggestions
        if topic != 'general':
            topic_data = self.topic_keywords[topic]
            missing_keywords = [kw for kw in topic_data['keywords'][:3] if kw not in analysis['keyword_matches']]
            if missing_keywords:
                feedback_parts.append(f"  • Consider mentioning: {', '.join(missing_keywords)}")
        
        return "\n".join(feedback_parts)
    
    def grade_answer(self, question: str, student_answer: str) -> Dict:
        """Grade a single answer"""
        
        # Identify topic
        topic = self.identify_topic(question)
        
        # Analyze answer content
        analysis = self.analyze_answer_content(student_answer, topic)
        
        # Calculate final score (0-5)
        raw_score = analysis['content_score']
        final_score = max(0, min(5, round(raw_score)))
        
        # Get RAG context if available
        rag_context = self.get_rag_context(question)
        
        # Generate feedback
        feedback = self.generate_feedback(question, student_answer, final_score, analysis, topic)
        
        return {
            'question': question,
            'student_answer': student_answer,
            'score': final_score,
            'max_score': 5,
            'topic': topic,
            'analysis': analysis,
            'feedback': feedback,
            'rag_context': rag_context,
            'timestamp': datetime.now().isoformat()
        }
    
    def grade_batch(self, questions_answers: List[Dict]) -> List[Dict]:
        """Grade multiple answers"""
        results = []
        
        print(f"Grading {len(questions_answers)} answers...")
        
        for i, qa in enumerate(questions_answers):
            print(f"Grading {i+1}/{len(questions_answers)}: {qa['question'][:50]}...")
            
            result = self.grade_answer(qa['question'], qa['student_answer'])
            results.append(result)
        
        return results
    
    def interactive_grading(self):
        """Interactive grading session"""
        print("🎓 Simple OS Answer Grading System")
        print("Enter 'quit' to exit")
        print("="*50)
        
        while True:
            print("\n" + "-"*50)
            question = input("Enter the question: ").strip()
            
            if question.lower() == 'quit':
                break
            
            student_answer = input("Enter the student's answer: ").strip()
            
            if student_answer.lower() == 'quit':
                break
            
            print("\nGrading...")
            result = self.grade_answer(question, student_answer)
            
            print(f"\n{'='*50}")
            print("GRADING RESULT")
            print(f"{'='*50}")
            print(f"Topic: {result['topic'].replace('_', ' ').title()}")
            print(f"Score: {result['score']}/5")
            print(f"\nFeedback:")
            print(result['feedback'])
            
            if result['rag_context']:
                print(f"\nRelevant Context:")
                print(result['rag_context'][:200] + "..." if len(result['rag_context']) > 200 else result['rag_context'])
            
            print(f"{'='*50}")

def main():
    """Main function"""
    print("🚀 Simple OS Answer Grading System")
    print("="*50)
    
    # Initialize grader
    grader = SimpleOSGrader(use_rag=True)
    
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
    
    print("\n📝 Testing with sample questions...")
    results = grader.grade_batch(test_questions)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}")
        print(f"{'='*60}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['student_answer']}")
        print(f"Topic: {result['topic'].replace('_', ' ').title()}")
        print(f"Score: {result['score']}/5")
        print(f"\nFeedback:")
        print(result['feedback'])
    
    # Generate summary report
    scores = [r['score'] for r in results]
    avg_score = sum(scores) / len(scores)
    
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total Questions: {len(results)}")
    print(f"Average Score: {avg_score:.2f}/5")
    print(f"Score Distribution: {dict(pd.Series(scores).value_counts().sort_index())}")
    
    # Save results
    with open("c:\\Users\\vinay\\Desktop\\research paper\\simple_grading_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to simple_grading_results.json")
    
    # Interactive session
    choice = input("\nWould you like to start an interactive grading session? (y/n): ")
    if choice.lower() == 'y':
        grader.interactive_grading()

if __name__ == "__main__":
    main()