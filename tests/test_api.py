#!/usr/bin/env python3
"""
API Testing Script for Rossmann Sales Forecasting Service
Machine Learning Zoomcamp 2025 - Midterm Project

Tests all API endpoints and validates responses.
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
                print(f"   Model loaded: {data.get('model_loaded')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
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
        """Test single prediction endpoint"""
        print("\n🔍 Testing prediction endpoint...")
        
        # Test case 1: Valid prediction
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
                
                # Validate prediction is reasonable
                if isinstance(sales, (int, float)) and sales > 0:
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
    
    def test_prediction_validation(self):
        """Test prediction endpoint input validation"""
        print("\n🔍 Testing prediction input validation...")
        
        # Test missing required fields
        invalid_data = {
            "Store": 1,
            # Missing Date and DayOfWeek
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data
            )
            
            if response.status_code == 400:
                print("✅ Input validation working correctly")
                return True
            else:
                print(f"❌ Input validation failed: Expected 400, got {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Validation test error: {str(e)}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
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
            ]
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
                
                print("✅ Batch prediction passed")
                print(f"   Total predictions: {total}")
                print(f"   Successful: {successful}")
                print(f"   Failed: {data.get('failed', 0)}")
                
                # Show first prediction
                if predictions:
                    first_pred = predictions[0]
                    print(f"   Sample prediction: {first_pred.get('prediction')}")
                
                return successful > 0
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            return False
    
    def test_error_handling(self):
        """Test API error handling"""
        print("\n🔍 Testing error handling...")
        
        # Test invalid endpoint
        try:
            response = self.session.get(f"{self.base_url}/invalid-endpoint")
            
            if response.status_code == 404:
                print("✅ 404 error handling working")
            else:
                print(f"❌ Expected 404, got {response.status_code}")
                return False
            
            # Test invalid method
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
        
        # Check if average response time is reasonable (< 1 second)
        if avg_time < 1.0:
            print("✅ Response times are acceptable")
            return True
        else:
            print("⚠️ Response times are slow (>1s)")
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
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20}: {status}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! API is ready for production.")
        else:
            print("⚠️ Some tests failed. Review issues before deployment.")
        
        return passed == total

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rossmann API")
    parser.add_argument(
        "--url", 
        default="http://localhost:5000",
        help="API base URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential tests"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.quick:
        # Quick test - just health and prediction
        print("🏃 Running quick tests...")
        health_ok = tester.test_health_endpoint()
        prediction_ok = tester.test_prediction_endpoint()
        
        if health_ok and prediction_ok:
            print("\n✅ Quick tests passed!")
        else:
            print("\n❌ Quick tests failed!")
    else:
        # Full test suite
        success = tester.run_all_tests()
        exit(0 if success else 1)

if __name__ == "__main__":
    main()