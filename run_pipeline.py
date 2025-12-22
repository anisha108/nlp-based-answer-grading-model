"""
Complete pipeline to run the OS Answer Grading System
This script orchestrates the entire process from data preprocessing to model training and evaluation
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        
        end_time = time.time()
        print(f"⏱️ Completed in {end_time - start_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout)
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("Checking requirements...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'pandas', 'numpy', 
        'scikit-learn', 'sentence-transformers', 'faiss-cpu', 
        'nltk', 'spacy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Installing missing packages...")
        
        install_cmd = f"pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing missing packages"):
            return False
    
    # Download spaCy model if needed
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model...")
        if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
            print("⚠️ Warning: spaCy model download failed, continuing without it")
    
    print("✅ All requirements satisfied")
    return True

def main():
    """Main pipeline execution"""
    
    print("🚀 OS Answer Grading System - Complete Pipeline")
    print(f"Started at: {datetime.now()}")
    
    # Change to project directory
    project_dir = "c:\\Users\\vinay\\Desktop\\research paper"
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please install missing packages manually.")
        return
    
    # Step 2: Data preprocessing
    if not run_command("python data_preprocessor.py", "Data Preprocessing"):
        print("❌ Data preprocessing failed. Check your CSV files.")
        return
    
    # Step 3: Build RAG knowledge base
    if not run_command("python rag_system.py", "Building RAG Knowledge Base"):
        print("❌ RAG system setup failed.")
        return
    
    # Step 4: Train the model
    print("\n⚠️ WARNING: Model training may take 30-60 minutes depending on your hardware")
    choice = input("Do you want to proceed with training? (y/n): ")
    
    if choice.lower() != 'y':
        print("Skipping model training. You can run it later with: python model_trainer.py")
    else:
        if not run_command("python model_trainer.py", "Training T5 Model"):
            print("❌ Model training failed.")
            return
    
    # Step 5: Test the grading system
    model_path = os.path.join(project_dir, "models", "best_model")
    if os.path.exists(model_path):
        print("\n🎯 Testing the grading system...")
        if not run_command("python grading_system.py", "Testing Grading System"):
            print("❌ Grading system test failed.")
            return
    else:
        print("⚠️ Trained model not found. Skipping grading system test.")
        print("Train the model first by running: python model_trainer.py")
    
    # Step 6: Generate summary
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("\n📁 Generated Files:")
    files_to_check = [
        "train_data.csv",
        "val_data.csv", 
        "test_data.csv",
        "knowledge_base.csv",
        "enhanced_train_data.csv",
        "models/best_model",
        "grading_report.json"
    ]
    
    for file_path in files_to_check:
        full_path = os.path.join(project_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print(f"\n📊 Next Steps:")
    print("1. Review the grading_report.json for model performance")
    print("2. Use grading_system.py for interactive grading")
    print("3. Integrate the system into your application")
    
    print(f"\n🔧 Usage Examples:")
    print("- Interactive grading: python grading_system.py")
    print("- Batch grading: Import OSGradingSystem class in your code")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()