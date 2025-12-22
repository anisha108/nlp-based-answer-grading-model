"""
Simple test to check if training dependencies work
"""

print("Testing training dependencies...")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

# Test data loading
try:
    train_df = pd.read_csv("train_data.csv")
    print(f"✅ Training data loaded: {len(train_df)} samples")
    
    val_df = pd.read_csv("val_data.csv")
    print(f"✅ Validation data loaded: {len(val_df)} samples")
    
    # Check data format
    required_cols = ['input_text', 'target_text', 'question', 'student_answer', 'score']
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")
        
        # Show sample
        print(f"\nSample training example:")
        print(f"Input: {train_df.iloc[0]['input_text'][:100]}...")
        print(f"Target: {train_df.iloc[0]['target_text'][:100]}...")
        print(f"Score: {train_df.iloc[0]['score']}")
        
except Exception as e:
    print(f"❌ Data loading error: {e}")

# Test model loading
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    print("\nTesting model loading...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    print("✅ T5-small model loaded successfully")
    
    # Test tokenization
    test_text = "Question: What is multithreading? Student Answer: Multithreading allows concurrent execution."
    tokens = tokenizer(test_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    print(f"✅ Tokenization works: {tokens['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ Model loading error: {e}")

print("\n" + "="*50)
print("DEPENDENCY CHECK COMPLETE")
print("="*50)