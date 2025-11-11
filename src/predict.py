#!/usr/bin/env python3
"""
Rossmann Store Sales Forecasting - Prediction Web Service
Machine Learning Zoomcamp 2025 - Midterm Project

Flask web service for serving sales predictions via REST API.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

class PredictionService:
    """Rossmann Sales Prediction Service"""
    
    def __init__(self, model_path="./models"):
        self.model_path = Path(model_path)
        self.model = None
        self.features = None
        self.store_info = None
        self.metadata = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all model artifacts"""
        try:
            print("🔄 Loading model artifacts...")
            
            # Load the trained model
            model_file = self.model_path / 'best_model.joblib'
            self.model = joblib.load(model_file)
            print(f"✅ Model loaded: {model_file}")
            
            # Load feature list
            features_file = self.model_path / 'feature_list.joblib'
            self.features = joblib.load(features_file)
            print(f"✅ Features loaded: {len(self.features)} features")
            
            # Load store information
            store_file = self.model_path / 'store_info.joblib'
            self.store_info = joblib.load(store_file)
            print(f"✅ Store info loaded: {len(self.store_info)} stores")
            
            # Load model metadata
            metadata_file = self.model_path / 'model_metadata.joblib'
            self.metadata = joblib.load(metadata_file)
            print(f"✅ Model metadata loaded: {self.metadata['model_name']}")
            
            print("🎉 All artifacts loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {str(e)}")
            raise
    
    def create_features(self, input_data):
        """Create features for prediction from input data"""
        
        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Merge with store information
        if 'Store' in df.columns:
            df = df.merge(self.store_info, on='Store', how='left')
        
        # Convert date and create time-based features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Cyclical encoding
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Competition features
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df['CompetitionDistance_log'] = np.log1p(df['CompetitionDistance'])
        
        # Competition open duration
        df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        df['CompetitionOpen'] = ((df['CompetitionOpenSinceYear'] > 0) & 
                               (df['CompetitionOpenSinceMonth'] > 0)).astype(int)
        
        # Promo2 features
        df['Promo2SinceWeek'].fillna(0, inplace=True)
        df['Promo2SinceYear'].fillna(0, inplace=True)
        df['PromoInterval'].fillna('None', inplace=True)
        
        # Encode categorical variables
        df['StoreType_encoded'] = df['StoreType'].astype('category').cat.codes
        df['Assortment_encoded'] = df['Assortment'].astype('category').cat.codes
        
        # Holiday interactions
        df['Holiday_Promo'] = df['SchoolHoliday'] * df['Promo']
        df['StateHoliday_binary'] = (df['StateHoliday'] != '0').astype(int)
        
        # Select only the features used in training
        feature_data = df[self.features]
        
        return feature_data
    
    def predict(self, input_data):
        """Make prediction for given input"""
        try:
            # Create features
            feature_data = self.create_features(input_data)
            
            # Make prediction
            prediction = self.model.predict(feature_data)
            
            # Convert to float for JSON serialization
            if len(prediction) == 1:
                return float(prediction[0])
            else:
                return [float(p) for p in prediction]
                
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': self.metadata['model_name'],
            'model_type': self.metadata['model_type'],
            'training_date': self.metadata['training_date'],
            'performance': self.metadata['performance'],
            'features_count': len(self.features),
            'version': '1.0.0'
        }

# Initialize prediction service
try:
    prediction_service = PredictionService()
except Exception as e:
    print(f"❌ Failed to initialize prediction service: {str(e)}")
    prediction_service = None

@app.route('/')
def home():
    """Home endpoint with service information"""
    return jsonify({
        'service': 'Rossmann Sales Forecasting API',
        'version': '1.0.0',
        'status': 'healthy' if prediction_service else 'error',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'info': '/info (GET)'
        },
        'description': 'Machine Learning Zoomcamp 2025 - Midterm Project'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    if prediction_service and prediction_service.model:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/info')
def info():
    """Model information endpoint"""
    if prediction_service:
        return jsonify(prediction_service.get_model_info())
    else:
        return jsonify({'error': 'Model not loaded'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if service is initialized
        if not prediction_service:
            return jsonify({'error': 'Prediction service not initialized'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Store', 'Date', 'DayOfWeek']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Set default values for optional fields
        defaults = {
            'Promo': 0,
            'SchoolHoliday': 0,
            'StateHoliday': '0'
        }
        
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
        
        # Make prediction
        prediction = prediction_service.predict(data)
        
        # Prepare response
        response = {
            'prediction': {
                'sales': round(prediction, 2),
                'confidence': 'high' if prediction > 1000 else 'medium'
            },
            'input': data,
            'model': {
                'name': prediction_service.metadata['model_name'],
                'version': '1.0.0'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if not prediction_service:
            return jsonify({'error': 'Prediction service not initialized'}), 500
        
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'No predictions data provided. Expected format: {\"predictions\": [...]}'})
        
        predictions_data = data['predictions']
        
        if not isinstance(predictions_data, list):
            return jsonify({'error': 'Predictions must be a list'}), 400
        
        if len(predictions_data) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size too large. Maximum 100 predictions at once.'}), 400
        
        results = []
        for i, item in enumerate(predictions_data):
            try:
                prediction = prediction_service.predict(item)
                results.append({
                    'index': i,
                    'input': item,
                    'prediction': round(prediction, 2),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'input': item,
                    'prediction': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error']),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Rossmann Sales Forecasting API")
    print("Machine Learning Zoomcamp 2025 - Midterm Project")
    print("=" * 60)
    
    # Check if model is loaded
    if prediction_service:
        print("✅ Prediction service initialized successfully")
        print(f"🤖 Model: {prediction_service.metadata['model_name']}")
        print(f"🎯 Performance: R² = {prediction_service.metadata['performance']['R²']:.4f}")
    else:
        print("❌ Prediction service failed to initialize")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🌐 API will be available at: http://localhost:{port}")
    print("\n📍 Available endpoints:")
    print("   GET  /          - Service information")
    print("   GET  /health    - Health check")
    print("   GET  /info      - Model information")
    print("   POST /predict   - Single prediction")
    print("   POST /predict/batch - Batch predictions")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)