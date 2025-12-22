"""
Simplified pipeline for OS Grading System
Handles missing dependencies gracefully and provides fallbacks
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import importlib

def check_dependency(package_name, import_name=None):
    """Check if a dependency is available"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def run_step(script_name, description, required=True):
    """Run a pipeline step"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"❌ Script {script_name} not found")
        return False
    
    try:
        print(f"Running: python {script_name}")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"  {line}")
            print(f"⏱️ Completed in {end_time - start_time:.2f} seconds")
            return True
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error output:")
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[-10:]:  # Show last 10 error lines
                    print(f"  {line}")
            if result.stdout:
                print("Standard output:")
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:  # Show last 5 output lines
                    print(f"  {line}")
            
            if required:
                return False
            else:
                print("⚠️ Optional step failed, continuing...")
                return True
                
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT - Step took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def check_system_status():
    """Check system status and available features"""
    print("🔍 Checking system status...")
    
    # Check core dependencies
    core_deps = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'transformers': 'transformers',
        'torch': 'torch'
    }
    
    optional_deps = {
        'sentence-transformers': 'sentence_transformers',
        'faiss-cpu': 'faiss',
        'scikit-learn': 'sklearn',
        'nltk': 'nltk'
    }
    
    core_available = all(check_dependency(pkg, imp) for pkg, imp in core_deps.items())
    optional_count = sum(1 for pkg, imp in optional_deps.items() if check_dependency(pkg, imp))
    
    print(f"Core dependencies: {'✅ Available' if core_available else '❌ Missing'}")
    print(f"Optional dependencies: {optional_count}/{len(optional_deps)} available")
    
    return core_available, optional_count

def main():
    """Main simplified pipeline"""
    print("🚀 OS Answer Grading System - Simplified Pipeline")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    # Change to project directory
    project_dir = "c:\\Users\\vinay\\Desktop\\research paper"
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Check system status
    core_available, optional_count = check_system_status()
    
    if not core_available:
        print("\n❌ Core dependencies missing!")
        print("Please run: python install_dependencies.py")
        choice = input("Do you want to try installing dependencies now? (y/n): ")
        if choice.lower() == 'y':
            if run_step("install_dependencies.py", "Installing Dependencies", required=False):
                core_available, optional_count = check_system_status()
        
        if not core_available:
            print("❌ Cannot proceed without core dependencies")
            return False
    
    print(f"\n✅ System ready! Core: Available, Optional: {optional_count}/4")
    
    # Step 1: Data preprocessing
    print("\n🔄 Starting pipeline steps...")
    if not run_step("data_preprocessor.py", "Data Preprocessing"):
        print("❌ Data preprocessing failed. Check your CSV files and try again.")
        return False
    
    # Step 2: RAG system (optional, may use fallback)
    print("\n📚 Building knowledge base...")
    rag_success = run_step("rag_system.py", "Building RAG Knowledge Base", required=False)
    if not rag_success:
        print("⚠️ RAG system failed, but continuing with basic functionality")
    
    # Step 3: Check if we should train the model
    print("\n🤖 Model training preparation...")
    model_path = os.path.join(project_dir, "models", "best_model")
    
    if os.path.exists(model_path):
        print("✅ Trained model found, skipping training")
        train_model = False
    else:
        print("⚠️ No trained model found")
        print("Model training requires significant time and computational resources:")
        print("  - Time: 30-60 minutes")
        print("  - Memory: 4-8 GB RAM")
        print("  - GPU recommended but not required")
        
        choice = input("Do you want to train the model now? (y/n): ")
        train_model = choice.lower() == 'y'
    
    if train_model:
        print("\n🏋️ Training model (this may take a while)...")
        if not run_step("model_trainer.py", "Training T5 Model"):
            print("❌ Model training failed")
            print("You can try again later or use a pre-trained model")
            return False
    
    # Step 4: Test the system
    if os.path.exists(model_path) or train_model:
        print("\n🧪 Testing the grading system...")
        if run_step("grading_system.py", "Testing Grading System", required=False):
            print("✅ Grading system test completed")
        else:
            print("⚠️ Grading system test had issues, but basic functionality may still work")
    else:
        print("\n⚠️ Skipping grading system test (no trained model)")
    
    # Step 5: Summary
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED!")
    print(f"{'='*60}")
    
    # Check what was created
    created_files = []
    expected_files = [
        ("train_data.csv", "Training data"),
        ("val_data.csv", "Validation data"),
        ("test_data.csv", "Test data"),
        ("knowledge_base.csv", "Knowledge base"),
        ("enhanced_train_data.csv", "Enhanced training data"),
        ("models/best_model", "Trained model"),
        ("grading_report.json", "Evaluation report")
    ]
    
    print("\n📁 Generated Files:")
    for filename, description in expected_files:
        filepath = os.path.join(project_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {description}: {filename}")
            created_files.append(filename)
        else:
            print(f"❌ {description}: {filename} (not created)")
    
    print(f"\n📊 Success Rate: {len(created_files)}/{len(expected_files)} files created")
    
    # Provide next steps
    print(f"\n🎯 What you can do now:")
    
    if "models/best_model" in created_files:
        print("✅ Full system ready!")
        print("  - Run: python grading_system.py (for interactive grading)")
        print("  - Use the OSGradingSystem class in your code")
        print("  - Check grading_report.json for performance metrics")
    elif "train_data.csv" in created_files:
        print("⚠️ Data processed but model not trained")
        print("  - Run: python model_trainer.py (to train the model)")
        print("  - Or use the demo grader: python demo.py")
    else:
        print("❌ Basic setup incomplete")
        print("  - Check your CSV files are in the correct format")
        print("  - Run: python demo.py (to test data loading)")
    
    print(f"\nCompleted at: {datetime.now()}")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n⚠️ Pipeline completed with issues.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
    
    input("\nPress Enter to exit...")