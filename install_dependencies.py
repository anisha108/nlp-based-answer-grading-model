"""
Dependency installer for OS Grading System
Handles installation of required packages with fallbacks
"""

import subprocess
import sys
import importlib
import os

def install_package(package_name, pip_name=None):
    """Install a package using pip"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        print(f"Installing {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", pip_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        print(f"❌ {package_name} is not available")
        return False

def main():
    """Main installation function"""
    print("🔧 Installing dependencies for OS Grading System")
    print("="*60)
    
    # Core dependencies (required)
    core_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("scikit-learn", "sklearn"),
        ("nltk", "nltk"),
        ("tqdm", "tqdm"),
    ]
    
    # Optional dependencies (for enhanced features)
    optional_packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("spacy", "spacy"),
        ("wandb", "wandb"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    print("Installing core dependencies...")
    core_success = True
    for package_name, import_name in core_packages:
        if not check_package(package_name, import_name):
            if not install_package(package_name):
                core_success = False
    
    print("\nInstalling optional dependencies...")
    optional_success = 0
    for package_name, import_name in optional_packages:
        if not check_package(package_name, import_name):
            if install_package(package_name):
                optional_success += 1
        else:
            optional_success += 1
    
    # Special handling for spaCy model
    if check_package("spacy"):
        try:
            import spacy
            try:
                spacy.load("en_core_web_sm")
                print("✅ spaCy English model is available")
            except OSError:
                print("📥 Downloading spaCy English model...")
                result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ spaCy English model downloaded successfully")
                else:
                    print("⚠️ Warning: Could not download spaCy model, will use fallback")
        except ImportError:
            pass
    
    print("\n" + "="*60)
    print("📊 INSTALLATION SUMMARY")
    print("="*60)
    
    if core_success:
        print("✅ Core dependencies: All installed successfully")
    else:
        print("❌ Core dependencies: Some packages failed to install")
    
    print(f"✅ Optional dependencies: {optional_success}/{len(optional_packages)} installed")
    
    print("\n🎯 System Status:")
    if core_success:
        print("✅ Basic functionality: Available")
        if optional_success >= 6:
            print("✅ Advanced features: Available (RAG, embeddings, visualization)")
        elif optional_success >= 3:
            print("⚠️ Advanced features: Partially available (some features may use fallbacks)")
        else:
            print("❌ Advanced features: Limited (will use basic fallbacks)")
    else:
        print("❌ System may not work properly due to missing core dependencies")
    
    print("\n📋 Next Steps:")
    if core_success:
        print("1. Run: python demo.py (to test the system)")
        print("2. Run: python data_preprocessor.py (to process your data)")
        print("3. Run: python rag_system.py (to build knowledge base)")
        print("4. Run: python model_trainer.py (to train the model)")
    else:
        print("1. Manually install missing core packages")
        print("2. Check your Python environment and pip installation")
        print("3. Consider using a virtual environment")
    
    return core_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n⚠️ Installation completed with some issues.")
    
    input("\nPress Enter to continue...")