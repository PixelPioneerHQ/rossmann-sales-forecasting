#!/usr/bin/env python3
"""
FastAPI Model Selection Demo Script
Machine Learning Zoomcamp 2025 - Midterm Project

Demonstrates the enhanced FastAPI with model selection capabilities.
"""

import requests
import json
import time
from datetime import datetime

def demo_fastapi_features(base_url="http://localhost:5000"):
    """Demonstrate all FastAPI features including model selection"""
    
    print("🚀 ROSSMANN FASTAPI MODEL SELECTION DEMO")
    print("=" * 60)
    print(f"🎯 Testing API at: {base_url}")
    print(f"📅 Demo started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test data
    sample_data = {
        "Store": 1,
        "Date": "2015-09-01", 
        "DayOfWeek": 2,
        "Promo": 1,
        "SchoolHoliday": 0
    }
    
    try:
        print("📖 1. INTERACTIVE DOCUMENTATION")
        print(f"   🌐 Visit: {base_url}/docs")
        print(f"   📋 ReDoc: {base_url}/redoc")
        print("   🎯 Features: Live testing, model selection, validation")
        print()
        
        print("🔍 2. HEALTH CHECK")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Status: {health_data.get('status')}")
            print(f"   🤖 Models loaded: {health_data.get('models_loaded', 0)}")
            print(f"   🥇 Default model: {health_data.get('default_model', 'Unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        print()
        
        print("📋 3. AVAILABLE MODELS")
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            print(f"   🎯 Available models: {models}")
            print(f"   🥇 Default model: {models_data.get('default_model')}")
            print(f"   📊 Total models: {models_data.get('total_models')}")
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
            return False
        print()
        
        print("🔍 4. MODEL INFORMATION")
        response = requests.get(f"{base_url}/info")
        if response.status_code == 200:
            info_data = response.json()
            print(f"   🤖 Model: {info_data.get('model_name')}")
            print(f"   🎯 Type: {info_data.get('model_type')}")
            print(f"   📊 Features: {info_data.get('features_count')}")
            performance = info_data.get('performance', {})
            if performance:
                print(f"   🏆 R² Score: {performance.get('R²', 'N/A')}")
                print(f"   📈 RMSE: {performance.get('RMSE', 'N/A')}")
        else:
            print(f"   ⚠️  Model info not available (models may need training)")
        print()
        
        print("🎯 5. MODEL SELECTION TESTING")
        
        # Test default model
        print("   📈 Testing default model prediction...")
        response = requests.post(
            f"{base_url}/predict",
            json=sample_data
        )
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', {})
            model_info = data.get('model', {})
            print(f"   ✅ Default prediction: {prediction.get('sales')}")
            print(f"   🤖 Model used: {model_info.get('name')}")
            print(f"   🎯 Model type: {model_info.get('type')}")
        elif response.status_code == 422:
            print("   📝 Pydantic validation working (422 response)")
            error_data = response.json()
            print(f"   📋 Validation details: {error_data}")
        else:
            print(f"   ⚠️  Default prediction failed: {response.status_code}")
        
        # Test specific model selection (if models are available)
        if models:
            print("   🔀 Testing model selection...")
            for model in models[:3]:  # Test first 3 models
                try:
                    response = requests.post(
                        f"{base_url}/predict?model={model}",
                        json=sample_data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get('prediction', {})
                        print(f"   ✅ {model}: {prediction.get('sales')}")
                    else:
                        print(f"   ⚠️  {model}: Status {response.status_code}")
                except Exception as e:
                    print(f"   ❌ {model}: Error - {str(e)}")
        print()
        
        print("📊 6. BATCH PREDICTION TEST")
        batch_data = {
            "predictions": [
                {"Store": 1, "Date": "2015-09-01", "DayOfWeek": 2, "Promo": 1},
                {"Store": 2, "Date": "2015-09-01", "DayOfWeek": 2, "Promo": 0}
            ],
            "model": "best"
        }
        
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Batch prediction successful")
            print(f"   📊 Total: {data.get('total')}")
            print(f"   ✅ Successful: {data.get('successful')}")
            print(f"   🤖 Model used: {data.get('model_used')}")
        else:
            print(f"   ⚠️  Batch prediction failed: {response.status_code}")
        print()
        
        print("⚡ 7. PERFORMANCE TEST")
        start_time = time.time()
        
        for i in range(5):
            response = requests.post(
                f"{base_url}/predict",
                json=sample_data
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        
        print(f"   ⚡ Average response time: {avg_time*1000:.1f}ms")
        print(f"   🎯 Performance: {'✅ Good' if avg_time < 1.0 else '⚠️ Acceptable' if avg_time < 2.0 else '❌ Slow'}")
        print()
        
        print("🎉 FASTAPI DEMO COMPLETED!")
        print("=" * 60)
        print("🌟 Key Features Demonstrated:")
        print("   📖 Automatic interactive documentation at /docs")
        print("   🔀 Model selection via query parameters")
        print("   📝 Pydantic validation for request/response")
        print("   📊 Comprehensive model information endpoints")
        print("   ⚡ High-performance async handling")
        print("   🛡️ Type safety and error handling")
        print()
        print("🚀 Next Steps:")
        print(f"   1. Open {base_url}/docs in your browser")
        print("   2. Try the interactive API documentation")
        print("   3. Test different models with the model parameter")
        print("   4. Run training: python src/train_fixed.py")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("💡 Make sure the FastAPI server is running:")
        print("   python src/predict.py")
        print()
        return False
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        return False

def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo FastAPI Model Selection Features")
    parser.add_argument(
        "--url",
        default="http://localhost:5000", 
        help="API base URL (default: http://localhost:5000)"
    )
    
    args = parser.parse_args()
    
    success = demo_fastapi_features(args.url)
    
    if success:
        print("\n🎯 Demo completed successfully!")
        print(f"🌐 Interactive docs: {args.url}/docs")
    else:
        print("\n❌ Demo failed. Check if the API server is running.")

if __name__ == "__main__":
    main()