#!/usr/bin/env python3
"""
Quick Test Script - Verify Model Loading and FastAPI Fixes
Machine Learning Zoomcamp 2025 - Midterm Project
"""

import sys
from pathlib import Path
import os

def test_model_files():
    """Test if model files exist in the expected location"""
    print("🔍 TESTING MODEL FILE LOCATIONS")
    print("=" * 50)
    
    models_dir = Path("models")
    print(f"Looking in: {models_dir.absolute()}")
    
    expected_files = [
        'linear_regression_model.joblib',
        'random_forest_model.joblib', 
        'xgboost_model.joblib',
        'arima_model.joblib',
        'best_model.joblib',
        'feature_list.joblib',
        'store_info.joblib',
        'model_metadata.joblib'
    ]
    
    missing_files = []
    found_files = []
    
    for file in expected_files:
        file_path = models_dir / file
        if file_path.exists():
            found_files.append(file)
            print(f"✅ Found: {file}")
        else:
            missing_files.append(file)
            print(f"❌ Missing: {file}")
    
    print(f"\n📊 Summary: {len(found_files)}/{len(expected_files)} files found")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All model files found!")
        return True

def test_predict_service_import():
    """Test if the prediction service can be imported without errors"""
    print("\n🔍 TESTING PREDICT SERVICE IMPORT")
    print("=" * 50)
    
    try:
        # Change to the src directory temporarily
        original_path = sys.path.copy()
        src_path = str(Path("src").absolute())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Try to import the prediction service
        from predict import MultiModelPredictionService
        print("✅ Import successful!")
        
        # Try to initialize (but don't load models yet)
        print("🔄 Testing service initialization...")
        service = MultiModelPredictionService()
        
        if hasattr(service, 'models') and hasattr(service, 'model_metadata'):
            print("✅ Service initialized successfully!")
            print(f"📊 Models found: {len(service.models)}")
            
            if service.models:
                print("✅ Models loaded successfully!")
                for model_name in service.models.keys():
                    print(f"   🤖 {model_name}")
                return True
            else:
                print("⚠️  No models loaded (but service works)")
                return True
        else:
            print("❌ Service initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        print("\nThis might be due to missing dependencies or path issues.")
        return False
    finally:
        # Restore original path
        sys.path = original_path

def main():
    """Run all tests"""
    print("🚀 QUICK DIAGNOSTIC TEST")
    print("Testing model loading and FastAPI fixes")
    print("=" * 60)
    
    # Test 1: Model files exist
    files_ok = test_model_files()
    
    # Test 2: Prediction service can import
    import_ok = test_predict_service_import()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Model files: {'✅ OK' if files_ok else '❌ FAILED'}")
    print(f"Service import: {'✅ OK' if import_ok else '❌ FAILED'}")
    
    if files_ok and import_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Model selection should now work correctly")
        print("\n🚀 Next steps:")
        print("   1. cd rossmann-sales-forecasting")
        print("   2. python src/predict.py")
        print("   3. Visit: http://localhost:5000/docs")
        print("   4. Test model selection with different models!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        if not files_ok:
            print("   🔧 Run: python src/train_fixed.py")
        if not import_ok:
            print("   🔧 Check dependencies and imports")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)