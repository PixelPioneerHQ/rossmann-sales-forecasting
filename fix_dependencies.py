#!/usr/bin/env python3
"""
Dependency Fix Script for Prophet + NumPy 2.0 Compatibility Issue
Machine Learning Zoomcamp 2025 - Midterm Project

This script fixes the NumPy 2.0 compatibility issue with Prophet.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚨 FIXING PROPHET + NUMPY 2.0 COMPATIBILITY ISSUE")
    print("=" * 60)
    print("Issue: Prophet 1.1.5 doesn't support NumPy 2.0+")
    print("Solution: Downgrade NumPy to 1.24.3")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: You're not in a virtual environment!")
        print("   It's recommended to use a virtual environment.")
        response = input("   Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Cancelled by user")
            return False
    else:
        print(f"✅ Virtual environment detected: {sys.prefix}")
    
    print()
    
    # Step 1: Uninstall problematic packages
    print("1. Uninstalling incompatible packages...")
    packages_to_remove = ["prophet", "numpy", "pandas", "scikit-learn"]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"Removing {package}")
    
    print()
    
    # Step 2: Install compatible versions
    print("2. Installing compatible versions...")
    
    compatible_packages = [
        "numpy==1.24.3",
        "pandas==2.1.4", 
        "scikit-learn==1.3.2",
        "prophet==1.1.5"
    ]
    
    for package in compatible_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            return False
    
    print()
    
    # Step 3: Install remaining requirements
    print("3. Installing remaining requirements...")
    if os.path.exists("requirements.txt"):
        if not run_command("pip install -r requirements.txt", "Installing all requirements"):
            print("❌ Failed to install requirements")
            return False
    else:
        print("⚠️  requirements.txt not found, skipping...")
    
    print()
    
    # Step 4: Verify installation
    print("4. Verifying installation...")
    
    try:
        import numpy as np
        import prophet
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ Prophet version: {prophet.__version__}")
        
        if np.__version__.startswith("1.24"):
            print("✅ NumPy version is compatible with Prophet!")
        else:
            print("⚠️  NumPy version might still be incompatible")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    print()
    print("🎉 DEPENDENCY FIX COMPLETED!")
    print("=" * 60)
    print("✅ NumPy downgraded to 1.24.3 (Prophet compatible)")
    print("✅ Prophet 1.1.5 should now work correctly")
    print("✅ All other dependencies updated")
    print()
    print("🚀 Next steps:")
    print("   1. python src/train_fixed.py  # Should work now!")
    print("   2. python src/predict.py      # Start the API")
    print("   3. python demo_fastapi.py     # Test model selection")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)