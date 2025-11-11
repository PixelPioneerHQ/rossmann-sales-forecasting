"""
Script to download Rossmann Store Sales dataset from Kaggle
Requires Kaggle API credentials to be set up
"""

import os
import zipfile
import pandas as pd
from pathlib import Path

def download_rossmann_data():
    """Download and extract Rossmann dataset from Kaggle"""
    
    # Check if Kaggle API is available
    try:
        import kaggle
        print("✓ Kaggle API found")
    except ImportError:
        print("❌ Kaggle package not found. Install with: pip install kaggle")
        return False
    
    # Set current directory as data folder
    data_dir = Path(__file__).parent
    os.chdir(data_dir)
    
    try:
        # Download the competition data
        print("📥 Downloading Rossmann Store Sales dataset...")
        kaggle.api.competition_download_files('rossmann-store-sales', path='.', quiet=False)
        
        # Extract the zip file
        zip_path = 'rossmann-store-sales.zip'
        if os.path.exists(zip_path):
            print("📦 Extracting dataset files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            # Clean up zip file
            os.remove(zip_path)
            print("🗑️ Cleaned up zip file")
            
        # Verify files exist
        required_files = ['train.csv', 'test.csv', 'store.csv']
        missing_files = []
        
        for file in required_files:
            if os.path.exists(file):
                # Get file info
                df = pd.read_csv(file)
                print(f"✓ {file}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            else:
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print("\n🎉 Dataset download completed successfully!")
        print("\nDataset structure:")
        print("- train.csv: Historical sales data (2013-2015)")
        print("- test.csv: Test period data for predictions") 
        print("- store.csv: Store attributes and metadata")
        print("\nNext: Run the Jupyter notebook for data exploration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Kaggle API credentials set up")
        print("2. Accept the competition rules at: https://www.kaggle.com/c/rossmann-store-sales")
        print("3. Check your internet connection")
        return False

def verify_data():
    """Verify that all required data files exist and are valid"""
    
    data_dir = Path(__file__).parent
    required_files = ['train.csv', 'test.csv', 'store.csv']
    
    print("🔍 Verifying dataset files...")
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"✓ {file}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"❌ Error reading {file}: {str(e)}")
                return False
        else:
            print(f"❌ Missing file: {file}")
            return False
    
    print("\n✅ All dataset files verified successfully!")
    return True

if __name__ == "__main__":
    print("🚀 Rossmann Dataset Downloader")
    print("=" * 50)
    
    # First check if data already exists
    if verify_data():
        print("\n📊 Dataset already exists and is valid!")
    else:
        print("\n📥 Downloading dataset from Kaggle...")
        download_rossmann_data()