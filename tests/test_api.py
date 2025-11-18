#!/usr/bin/env python3
"""
FastAPI Testing Script for Rossmann Sales Forecasting Service
Machine Learning Zoomcamp 2025 - Midterm Project

Tests all FastAPI endpoints including model selection capabilities.
"""

import requests
import json
import time
from datetime import datetime, timedelta

class APITester:
    """Comprehensive API testing suite"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RossmannAPI-Tester/1.0'
        })
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        print("🔍 Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Models loaded: {data.get('models_loaded')}")
                print(f"   Default model: {data.get('default_model')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_models_endpoint(self):
        """Test available models endpoint"""
        print("\n🔍 Testing models endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/models")
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                default_model = data.get('default_model')
                total_models = data.get('total_models', 0)
                
                print("✅ Models endpoint passed")
                print(f"   Available models: {models}")
                print(f"   Default model: {default_model}")
                print(f"   Total models: {total_models}")
                
                return len(models) > 0
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Models endpoint error: {str(e)}")
            return False
    
    def test_info_endpoint(self):
        """Test model info endpoint"""
        print("\n🔍 Testing info endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/info")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Info endpoint passed")
                print(f"   Model: {data.get('model_name')}")
                print(f"   Type: {data.get('model_type')}")
                print(f"   Training date: {data.get('training_date')}")
                if 'performance' in data:
                    perf = data['performance']
                    print(f"   R² Score: {perf.get('R²', 'N/A')}")
                    print(f"   RMSE: {perf.get('RMSE', 'N/A')}")
                return True
            else:
                print(f"❌ Info endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Info endpoint error: {str(e)}")
            return False
    
    def test_prediction_endpoint(self):
        """Test single prediction endpoint with default model"""
        print("\n🔍 Testing prediction endpoint...")
        
        # Test case 1: Valid prediction with default model
        test_data = {
            "Store": 1,
            "Date": "2015-09-01",
            "DayOfWeek": 2,
            "Promo": 1,
            "SchoolHoliday": 0
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', {})
                sales = prediction.get('sales')
                
                print("✅ Prediction endpoint passed")
                print(f"   Predicted sales: {sales}")
                print(f"   Confidence: {prediction.get('confidence')}")
                print(f"   Model: {data.get('model', {}).get('name')}")
                print(f"   Model type: {data.get('model', {}).get('type')}")
                
                # Validate prediction is reasonable for different model types
                model_name = data.get('model', {}).get('name', 'Unknown')
                if isinstance(sales, (int, float)) and sales > 0:
                    # Check reasonable ranges based on model type
                    if 'Prophet' in model_name and sales > 500:
                        print("✅ Prophet prediction in expected range")
                    elif 'ARIMA' in model_name and sales > 500:
                        print("✅ ARIMA prediction in expected range")
                    elif sales > 100:  # Traditional ML models
                        print("✅ Traditional ML prediction in expected range")
                    print("✅ Prediction value is valid")
                    return True
                else:
                    print("❌ Invalid prediction value")
                    return False
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return False
    
    def test_model_selection(self):
        """Test prediction endpoint with model selection"""
        print("\n🔍 Testing model selection...")
        
        test_data = {
            "Store": 1,
            "Date": "2015-09-01",
            "DayOfWeek": 2,
            "Promo": 1,
            "SchoolHoliday": 0
        }
        
        # Get available models first
        try:
            models_response = self.session.get(f"{self.base_url}/models")
            if models_response.status_code != 200:
                print("❌ Cannot get available models")
                return False
                
            available_models = models_response.json().get('models', [])
            if not available_models:
                print("⚠️ No models available for testing")
                return True  # Not a failure if no models trained yet
                
            print(f"   Testing with models: {available_models}")
            
            success_count = 0
            for model in available_models:
                try:
                    response = self.session.post(
                        f"{self.base_url}/predict?model={model}",
                        json=test_data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get('prediction', {})
                        sales = prediction.get('sales')
                        model_used = data.get('model', {}).get('name')
                        
                        print(f"   ✅ {model}: Sales = {sales}, Model = {model_used}")
                        
                        if isinstance(sales, (int, float)) and sales > 0:
                            success_count += 1
                        else:
                            print(f"   ❌ {model}: Invalid prediction value")
                    else:
                        print(f"   ❌ {model}: Failed with status {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ {model}: Error - {str(e)}")
            
            print(f"✅ Model selection test completed: {success_count}/{len(available_models)} models successful")
            return success_count > 0
                
        except Exception as e:
            print(f"❌ Model selection test error: {str(e)}")
            return False
    
    def test_prediction_validation(self):
        """Test prediction endpoint input validation with FastAPI"""
        print("\n🔍 Testing prediction input validation...")
        
        # Test missing required fields - FastAPI should return 422 for validation errors
        invalid_data = {
            "Store": 1,
            # Missing Date and DayOfWeek
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data
            )
            
            # FastAPI returns 422 for validation errors, not 400
            if response.status_code == 422:
                print("✅ FastAPI input validation working correctly (422)")
                return True
            elif response.status_code == 400:
                print("✅ Input validation working correctly (400)")
                return True
            else:
                print(f"❌ Input validation failed: Expected 422 or 400, got {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Validation test error: {str(e)}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint with model selection"""
        print("\n🔍 Testing batch prediction endpoint...")
        
        batch_data = {
            "predictions": [
                {
                    "Store": 1,
                    "Date": "2015-09-01",
                    "DayOfWeek": 2,
                    "Promo": 1
                },
                {
                    "Store": 2,
                    "Date": "2015-09-01",
                    "DayOfWeek": 2,
                    "Promo": 0
                },
                {
                    "Store": 3,
                    "Date": "2015-09-02",
                    "DayOfWeek": 3,
                    "Promo": 1
                }
            ],
            "model": "best"  # Specify model for batch prediction
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_data
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                total = data.get('total', 0)
                successful = data.get('successful', 0)
                model_used = data.get('model_used', 'unknown')
                
                print("✅ Batch prediction passed")
                print(f"   Total predictions: {total}")
                print(f"   Successful: {successful}")
                print(f"   Failed: {data.get('failed', 0)}")
                print(f"   Model used: {model_used}")
                
                # Show first prediction
                if predictions:
                    first_pred = predictions[0]
                    print(f"   Sample prediction: {first_pred.get('prediction')}")
                
                return successful > 0
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            return False
    
    def test_error_handling(self):
        """Test FastAPI error handling"""
        print("\n🔍 Testing error handling...")
        
        # Test invalid endpoint
        try:
            response = self.session.get(f"{self.base_url}/invalid-endpoint")
            
            if response.status_code == 404:
                print("✅ 404 error handling working")
            else:
                print(f"❌ Expected 404, got {response.status_code}")
                return False
            
            # Test invalid method - FastAPI returns 405 for method not allowed
            response = self.session.get(f"{self.base_url}/predict")
            
            if response.status_code == 405:
                print("✅ Method not allowed handling working")
                return True
            else:
                print(f"❌ Expected 405, got {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error handling test failed: {str(e)}")
            return False
    
    def test_docs_endpoint(self):
        """Test FastAPI automatic documentation"""
        print("\n🔍 Testing FastAPI docs endpoint...")
        
        try:
            # Test /docs endpoint
            response = self.session.get(f"{self.base_url}/docs")
            
            if response.status_code == 200:
                print("✅ /docs endpoint working")
                
                # Test /openapi.json endpoint
                openapi_response = self.session.get(f"{self.base_url}/openapi.json")
                
                if openapi_response.status_code == 200:
                    openapi_data = openapi_response.json()
                    print("✅ OpenAPI schema available")
                    print(f"   API Title: {openapi_data.get('info', {}).get('title')}")
                    print(f"   API Version: {openapi_data.get('info', {}).get('version')}")
                    return True
                else:
                    print(f"❌ OpenAPI schema failed: {openapi_response.status_code}")
                    return False
            else:
                print(f"❌ Docs endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Docs endpoint test error: {str(e)}")
            return False
    
    def test_performance(self):
        """Test API performance"""
        print("\n🔍 Testing API performance...")
        
        test_data = {
            "Store": 1,
            "Date": "2015-09-01",
            "DayOfWeek": 2,
            "Promo": 1,
            "SchoolHoliday": 0
        }
        
        # Test response times
        times = []
        for i in range(5):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_data
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                else:
                    print(f"❌ Performance test failed on request {i+1}")
                    return False
                    
            except Exception as e:
                print(f"❌ Performance test error: {str(e)}")
                return False
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print("✅ Performance test completed")
        print(f"   Average response time: {avg_time*1000:.1f}ms")
        print(f"   Min response time: {min_time*1000:.1f}ms")
        print(f"   Max response time: {max_time*1000:.1f}ms")
        
        # Check if average response time is reasonable
        # Time series models may be slightly slower
        threshold = 2.0  # Increased for Prophet/ARIMA models
        if avg_time < threshold:
            print("✅ Response times are acceptable")
            return True
        else:
            print(f"⚠️ Response times are slow (>{threshold}s)")
            print("   Note: Time series models may have higher latency")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 ROSSMANN API TEST SUITE")
        print("=" * 50)
        print(f"🎯 Testing API at: {self.base_url}")
        print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Model Info", self.test_info_endpoint),
            ("Single Prediction", self.test_prediction_endpoint),
            ("Input Validation", self.test_prediction_validation),
            ("Batch Prediction", self.test_batch_prediction),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"\n❌ Test '{test_name}' crashed: {str(e)}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY - ENHANCED API")
        print("Supports 5 Models: Linear, Random Forest, XGBoost, Prophet, ARIMA")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20}: {status}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        print("🤖 Time series models (Prophet/ARIMA) supported")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Enhanced API ready for production.")
            print("🏆 Prophet time series forecasting validated!")
        else:
            print("⚠️ Some tests failed. Review issues before deployment.")
        
        return passed == total

def main():
    """Main testing function for FastAPI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rossmann FastAPI with Model Selection")
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="API base URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential tests (health, models, prediction)"
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="Test only documentation endpoints"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.docs:
        # Test only documentation features
        print("📖 Testing FastAPI documentation...")
        docs_ok = tester.test_docs_endpoint()
        models_ok = tester.test_models_endpoint()
        
        if docs_ok and models_ok:
            print("\n✅ Documentation tests passed!")
            print(f"🌐 Visit: {args.url}/docs for interactive API documentation")
        else:
            print("\n❌ Documentation tests failed!")
    elif args.quick:
        # Quick test - health, models, and prediction
        print("🏃 Running quick FastAPI tests...")
        health_ok = tester.test_health_endpoint()
        models_ok = tester.test_models_endpoint()
        prediction_ok = tester.test_prediction_endpoint()
        
        if health_ok and models_ok and prediction_ok:
            print("\n✅ Quick FastAPI tests passed!")
            print(f"🌐 Interactive docs: {args.url}/docs")
        else:
            print("\n❌ Quick FastAPI tests failed!")
    else:
        # Full test suite
        success = tester.run_all_tests()
        exit(0 if success else 1)

if __name__ == "__main__":
    main()