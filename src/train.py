#!/usr/bin/env python3
"""
Rossmann Store Sales Forecasting - Model Training Script
Machine Learning Zoomcamp 2025 - Midterm Project

This script trains the final model for Rossmann sales prediction.
It implements the complete training pipeline from data loading to model saving.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

class RossmannSalesPredictor:
    """Rossmann Sales Forecasting Model Training Pipeline"""
    
    def __init__(self, data_path="../data", model_path="../src/models"):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.results = []
        
        print("🚀 Rossmann Sales Predictor initialized")
        print(f"📁 Data path: {self.data_path.absolute()}")
        print(f"💾 Model path: {self.model_path.absolute()}")
    
    def load_data(self):
        """Load and merge all datasets"""
        print("\n📊 Loading datasets...")
        
        # Load datasets
        train_df = pd.read_csv(self.data_path / 'train.csv')
        store_df = pd.read_csv(self.data_path / 'store.csv')
        
        print(f"✅ Train data: {train_df.shape}")
        print(f"✅ Store data: {store_df.shape}")
        
        # Merge datasets
        self.data = train_df.merge(store_df, on='Store', how='left')
        print(f"✅ Merged data: {self.data.shape}")
        
        # Basic data info
        print(f"📅 Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"🏪 Stores: {self.data['Store'].nunique()}")
        print(f"💰 Total sales: {self.data['Sales'].sum():,.0f}")
        
        return self.data
    
    def create_features(self, df):
        """Create comprehensive feature set for modeling"""
        print("\n⚙️ Engineering features...")
        
        df = df.copy()
        
        # Convert date and create time-based features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Cyclical encoding for seasonal patterns
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
        
        print(f"✅ Features created: {df.shape[1]} total columns")
        return df
    
    def prepare_model_data(self):
        """Prepare data for machine learning models"""
        print("\n🎯 Preparing model data...")
        
        # Create features
        self.data_featured = self.create_features(self.data)
        
        # Filter for open stores with non-zero sales
        model_data = self.data_featured[
            (self.data_featured['Open'] == 1) & 
            (self.data_featured['Sales'] > 0)
        ].copy()
        
        print(f"📊 Model training data: {model_data.shape}")
        print(f"💰 Sales range: {model_data['Sales'].min():,.0f} - {model_data['Sales'].max():,.0f}")
        
        # Define feature columns
        self.feature_columns = [
            'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
            'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter', 'DayOfYear',
            'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'StoreType_encoded', 'Assortment_encoded',
            'CompetitionDistance_log', 'CompetitionOpen',
            'Promo2', 'Holiday_Promo', 'StateHoliday_binary'
        ]
        
        # Ensure all features exist
        available_features = [col for col in self.feature_columns if col in model_data.columns]
        missing_features = set(self.feature_columns) - set(available_features)
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            self.feature_columns = available_features
        
        print(f"✅ Using {len(self.feature_columns)} features")
        
        # Prepare X and y
        X = model_data[self.feature_columns]
        y = model_data['Sales']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"📊 Training set: {self.X_train.shape}")
        print(f"📊 Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Calculate comprehensive model performance metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        results = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
        
        return results
    
    def train_linear_regression(self):
        """Train Linear Regression baseline model"""
        print("\n🔵 Training Linear Regression (Baseline)...")
        
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        
        lr_pred = lr_model.predict(self.X_test)
        lr_results = self.evaluate_model(self.y_test, lr_pred, 'Linear Regression')
        
        self.models['Linear Regression'] = lr_model
        self.results.append(lr_results)
        
        print(f"✅ Linear Regression - RMSE: {lr_results['RMSE']:.2f}, R²: {lr_results['R²']:.4f}")
        return lr_model, lr_results
    
    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n🌳 Training Random Forest...")
        
        # Initial model with good defaults
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)
        rf_results = self.evaluate_model(self.y_test, rf_pred, 'Random Forest')
        
        self.models['Random Forest'] = rf_model
        self.results.append(rf_results)
        
        print(f"✅ Random Forest - RMSE: {rf_results['RMSE']:.2f}, R²: {rf_results['R²']:.4f}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return rf_model, rf_results
    
    def train_xgboost(self):
        """Train XGBoost model with optimization"""
        print("\n🚀 Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        xgb_pred = xgb_model.predict(self.X_test)
        xgb_results = self.evaluate_model(self.y_test, xgb_pred, 'XGBoost')
        
        self.models['XGBoost'] = xgb_model
        self.results.append(xgb_results)
        
        print(f"✅ XGBoost - RMSE: {xgb_results['RMSE']:.2f}, R²: {xgb_results['R²']:.4f}")
        return xgb_model, xgb_results
    
    def select_best_model(self):
        """Select the best performing model based on R² score"""
        print("\n🏆 MODEL SELECTION")
        print("=" * 50)
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.round(4)
        
        print("📊 Model Comparison:")
        print(results_df.to_string(index=False))
        
        # Select best model by R² score
        best_idx = results_df['R²'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🥇 BEST MODEL: {self.best_model_name}")
        print(f"   R² Score: {results_df.loc[best_idx, 'R²']:.4f}")
        print(f"   RMSE: {results_df.loc[best_idx, 'RMSE']:.2f}")
        print(f"   MAPE: {results_df.loc[best_idx, 'MAPE']:.2f}%")
        
        return self.best_model, self.best_model_name, results_df
    
    def save_model_artifacts(self):
        """Save model and related artifacts for deployment"""
        print("\n💾 Saving model artifacts...")
        
        # Save the best model
        model_path = self.model_path / 'best_model.joblib'
        joblib.dump(self.best_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save feature list
        feature_list_path = self.model_path / 'feature_list.joblib'
        joblib.dump(self.feature_columns, feature_list_path)
        print(f"✅ Feature list saved: {feature_list_path}")
        
        # Save store information
        store_df = pd.read_csv(self.data_path / 'store.csv')
        store_info_path = self.model_path / 'store_info.joblib'
        joblib.dump(store_df, store_info_path)
        print(f"✅ Store info saved: {store_info_path}")
        
        # Save model metadata
        results_df = pd.DataFrame(self.results)
        best_idx = results_df['R²'].idxmax()
        
        model_metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'features': self.feature_columns,
            'performance': results_df.loc[best_idx].to_dict(),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'all_results': results_df.to_dict('records')
        }
        
        metadata_path = self.model_path / 'model_metadata.joblib'
        joblib.dump(model_metadata, metadata_path)
        print(f"✅ Model metadata saved: {metadata_path}")
        
        print(f"\n🎉 All artifacts saved in: {self.model_path.absolute()}")
    
    def train_pipeline(self):
        """Complete training pipeline"""
        print("🚀 ROSSMANN SALES FORECASTING - TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            # Load and prepare data
            self.load_data()
            self.prepare_model_data()
            
            # Train models
            self.train_linear_regression()
            self.train_random_forest()
            self.train_xgboost()
            
            # Select best model and save artifacts
            self.select_best_model()
            self.save_model_artifacts()
            
            print("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"🏆 Best model: {self.best_model_name}")
            print(f"📁 Models saved in: {self.model_path.absolute()}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            return False

def main():
    """Main training function"""
    print("🎯 Rossmann Sales Forecasting - Model Training")
    print("Machine Learning Zoomcamp 2025 - Midterm Project")
    print("=" * 60)
    
    # Initialize and run training pipeline
    predictor = RossmannSalesPredictor()
    success = predictor.train_pipeline()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("✅ Ready for deployment with Flask web service")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        exit(1)

if __name__ == "__main__":
    main()