import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
import json

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss-cpu not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

class RAGKnowledgeBase:
    """
    Retrieval-Augmented Generation system for OS textbook knowledge
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize RAG system
        
        Args:
            model_name: SentenceTransformer model for embeddings
        """
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        self.is_built = False
        self.use_simple_matching = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(model_name)
                print(f"Initialized RAG with SentenceTransformer: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer: {e}")
                print("Falling back to simple text matching")
                self.use_simple_matching = True
                self.encoder = None
        else:
            print("Missing dependencies. Using simple text matching instead of embeddings.")
            self.use_simple_matching = True
            self.encoder = None
    
    def build_knowledge_base(self, knowledge_df: pd.DataFrame):
        """
        Build the knowledge base from textbook content
        
        Args:
            knowledge_df: DataFrame with textbook content
        """
        print("Building knowledge base...")
        
        # Extract text chunks
        self.knowledge_base = []
        texts = []
        
        for idx, row in knowledge_df.iterrows():
            text = row.get('text', '').strip()
            if text and len(text) > 20:  # Filter out very short texts
                entry = {
                    'id': idx,
                    'text': text,
                    'topic': row.get('topic', 'general'),
                    'source': row.get('source', f'chunk_{idx}')
                }
                self.knowledge_base.append(entry)
                texts.append(text)
        
        if not texts:
            print("Warning: No valid texts found in knowledge base")
            return
        
        if self.use_simple_matching:
            # Simple text-based matching fallback
            print(f"Building simple text-based knowledge base with {len(texts)} chunks...")
            self.is_built = True
            print(f"Knowledge base built with {len(self.knowledge_base)} entries (simple matching)")
        else:
            # Generate embeddings
            print(f"Generating embeddings for {len(texts)} text chunks...")
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
            
            # Build FAISS index
            print("Building FAISS index...")
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
            
            self.is_built = True
            print(f"Knowledge base built with {len(self.knowledge_base)} entries")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Question or text to find relevant context for
            top_k: Number of top results to return
            
        Returns:
            List of relevant context entries
        """
        if not self.is_built:
            print("Warning: Knowledge base not built yet")
            return []
        
        if self.use_simple_matching:
            return self._simple_text_matching(query, top_k)
        else:
            # Encode query
            query_embedding = self.encoder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Retrieve results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_base):
                    result = self.knowledge_base[idx].copy()
                    result['relevance_score'] = float(score)
                    results.append(result)
            
            return results
    
    def _simple_text_matching(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple text-based matching fallback
        """
        query_words = set(query.lower().split())
        scored_entries = []
        
        for entry in self.knowledge_base:
            text_words = set(entry['text'].lower().split())
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / len(query_words.union(text_words))
                entry_copy = entry.copy()
                entry_copy['relevance_score'] = score
                scored_entries.append(entry_copy)
        
        # Sort by score and return top_k
        scored_entries.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_entries[:top_k]
    
    def get_context_for_question(self, question: str, max_context_length: int = 1000) -> str:
        """
        Get relevant context for a specific question
        
        Args:
            question: The question to find context for
            max_context_length: Maximum length of context to return
            
        Returns:
            Concatenated relevant context
        """
        relevant_entries = self.retrieve_relevant_context(question, top_k=5)
        
        context_parts = []
        current_length = 0
        
        for entry in relevant_entries:
            text = entry['text']
            if current_length + len(text) <= max_context_length:
                context_parts.append(text)
                current_length += len(text)
            else:
                # Add partial text if it fits
                remaining_space = max_context_length - current_length
                if remaining_space > 50:  # Only add if meaningful space left
                    context_parts.append(text[:remaining_space] + "...")
                break
        
        return " ".join(context_parts)
    
    def save_knowledge_base(self, save_path: str):
        """Save the knowledge base to disk"""
        if not self.is_built:
            print("Warning: Knowledge base not built yet")
            return
        
        save_data = {
            'knowledge_base': self.knowledge_base,
            'use_simple_matching': self.use_simple_matching,
            'embeddings': self.embeddings if not self.use_simple_matching else None,
        }
        
        # Save FAISS index only if using embeddings
        if not self.use_simple_matching and self.index is not None:
            faiss.write_index(self.index, f"{save_path}_faiss.index")
        
        # Save other data
        with open(f"{save_path}_data.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Knowledge base saved to {save_path}")
    
    def load_knowledge_base(self, load_path: str):
        """Load the knowledge base from disk"""
        try:
            # Load other data
            with open(f"{load_path}_data.pkl", 'rb') as f:
                save_data = pickle.load(f)
            
            self.knowledge_base = save_data['knowledge_base']
            self.use_simple_matching = save_data.get('use_simple_matching', False)
            self.embeddings = save_data.get('embeddings')
            
            # Load FAISS index only if not using simple matching
            if not self.use_simple_matching and os.path.exists(f"{load_path}_faiss.index"):
                self.index = faiss.read_index(f"{load_path}_faiss.index")
            
            self.is_built = True
            
            matching_type = "simple text matching" if self.use_simple_matching else "embeddings"
            print(f"Knowledge base loaded from {load_path} (using {matching_type})")
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")

class ContextEnhancedDataset:
    """
    Dataset class that enhances questions with relevant context using RAG
    """
    
    def __init__(self, data_df: pd.DataFrame, rag_system: RAGKnowledgeBase):
        self.data_df = data_df
        self.rag_system = rag_system
    
    def enhance_with_context(self, max_context_length: int = 800) -> pd.DataFrame:
        """
        Enhance dataset with relevant context from RAG system
        
        Args:
            max_context_length: Maximum length of context to add
            
        Returns:
            Enhanced DataFrame
        """
        enhanced_data = []
        
        print("Enhancing dataset with RAG context...")
        
        for idx, row in self.data_df.iterrows():
            question = row.get('question', '')
            
            # Get relevant context
            if self.rag_system.is_built and question:
                rag_context = self.rag_system.get_context_for_question(
                    question, max_context_length
                )
                
                # Combine existing context with RAG context
                existing_context = row.get('context', '')
                if existing_context and rag_context:
                    combined_context = f"{existing_context} {rag_context}"
                elif rag_context:
                    combined_context = rag_context
                else:
                    combined_context = existing_context
            else:
                combined_context = row.get('context', '')
            
            # Create enhanced input text
            enhanced_input = f"Question: {question}\nStudent Answer: {row.get('student_answer', '')}\nContext: {combined_context}\nTask: Grade and provide feedback"
            
            enhanced_row = row.copy()
            enhanced_row['enhanced_context'] = combined_context
            enhanced_row['input_text'] = enhanced_input
            
            enhanced_data.append(enhanced_row)
        
        enhanced_df = pd.DataFrame(enhanced_data)
        print(f"Dataset enhanced with RAG context")
        
        return enhanced_df

def main():
    """Main function to build and test RAG system"""
    
    # Load knowledge base
    try:
        kb_df = pd.read_csv("c:\\Users\\vinay\\Desktop\\research paper\\knowledge_base.csv")
        print(f"Loaded knowledge base with {len(kb_df)} entries")
    except FileNotFoundError:
        print("Knowledge base not found. Please run data_preprocessor.py first.")
        return
    
    # Initialize RAG system
    rag_system = RAGKnowledgeBase()
    
    # Build knowledge base
    rag_system.build_knowledge_base(kb_df)
    
    # Save knowledge base
    rag_system.save_knowledge_base("c:\\Users\\vinay\\Desktop\\research paper\\rag_kb")
    
    # Test retrieval
    test_questions = [
        "What is multithreading?",
        "How does kernel mode work?",
        "Explain memory management",
        "What is process scheduling?"
    ]
    
    print("\nTesting RAG retrieval:")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        context = rag_system.get_context_for_question(question, max_context_length=200)
        print(f"Retrieved context: {context[:200]}...")
    
    # Enhance training data
    try:
        train_df = pd.read_csv("c:\\Users\\vinay\\Desktop\\research paper\\train_data.csv")
        
        context_enhancer = ContextEnhancedDataset(train_df, rag_system)
        enhanced_train_df = context_enhancer.enhance_with_context()
        
        # Save enhanced data
        enhanced_train_df.to_csv("c:\\Users\\vinay\\Desktop\\research paper\\enhanced_train_data.csv", index=False)
        
        print(f"\nEnhanced training data saved with {len(enhanced_train_df)} samples")
        
    except FileNotFoundError:
        print("Training data not found. Please run data_preprocessor.py first.")

if __name__ == "__main__":
    main()