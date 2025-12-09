"""
Phase 4: Model Deployment for AMR Analysis

This module provides deployment functionality for trained AMR models:
1. Load saved models and make predictions on new data
2. Batch prediction from CSV files
3. Output predictions with probabilities and metadata

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings


class ModelDeployment:
    """
    Deployment class for trained AMR models.
    
    Handles loading models, preprocessing new data, and making predictions.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize deployment with a trained model.
        
        Args:
            model_path: Path to saved model file (.pkl)
        """
        self.model_path = model_path
        self.pipeline = None
        self.metadata = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained pipeline and metadata."""
        # Load pipeline
        self.pipeline = joblib.load(self.model_path)
        print(f"Model loaded from: {self.model_path}")
        
        # Load metadata
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Metadata loaded from: {metadata_path}")
            print(f"  Task: {self.metadata['model_info']['task_name']}")
            print(f"  Model: {self.metadata['model_info']['model_type']}")
            print(f"  Features: {self.metadata['features']['num_features']}")
        else:
            warnings.warn(f"Metadata file not found: {metadata_path}")
    
    def get_required_features(self) -> List[str]:
        """
        Get list of required feature columns.
        
        Returns:
            List of feature column names
        """
        if self.metadata is None:
            raise ValueError("No metadata available. Cannot determine required features.")
        return self.metadata['features']['feature_columns']
    
    def get_model_info(self) -> Dict:
        """
        Get model information from metadata.
        
        Returns:
            Dictionary with model information
        """
        if self.metadata is None:
            return {'warning': 'No metadata available'}
        return self.metadata['model_info']
    
    def get_performance_metrics(self) -> Dict:
        """
        Get model performance metrics from metadata.
        
        Returns:
            Dictionary with train/val/test metrics
        """
        if self.metadata is None:
            return {'warning': 'No metadata available'}
        return self.metadata['metrics']
    
    def predict(
        self,
        X: pd.DataFrame,
        include_proba: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            X: DataFrame with feature columns
            include_proba: Whether to include prediction probabilities
            
        Returns:
            predictions (and probabilities if include_proba=True)
        """
        # Verify features
        required_features = self.get_required_features()
        missing_features = set(required_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only required features in correct order
        X_selected = X[required_features]
        
        # Make predictions
        predictions = self.pipeline.predict(X_selected)
        
        if include_proba:
            if hasattr(self.pipeline, 'predict_proba'):
                probabilities = self.pipeline.predict_proba(X_selected)
                return predictions, probabilities
            else:
                warnings.warn("Model does not support probability predictions")
                return predictions, None
        
        return predictions
    
    def predict_from_csv(
        self,
        input_csv: str,
        output_csv: str,
        include_proba: bool = True,
        include_original: bool = True
    ) -> pd.DataFrame:
        """
        Predict from CSV file and save results.
        
        Args:
            input_csv: Path to input CSV file with new isolates
            output_csv: Path to save output CSV with predictions
            include_proba: Whether to include prediction probabilities
            include_original: Whether to include original columns in output
            
        Returns:
            DataFrame with predictions
        """
        print(f"\nReading data from: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"  Loaded {len(df)} samples")
        
        # Check for required features
        required_features = self.get_required_features()
        missing_features = set(required_features) - set(df.columns)
        
        if missing_features:
            raise ValueError(
                f"Input data is missing required features:\n"
                f"  Missing: {sorted(missing_features)}\n"
                f"  Required: {len(required_features)} features\n"
                f"  Available: {len(df.columns)} columns"
            )
        
        print(f"\nMaking predictions...")
        
        # Make predictions
        if include_proba:
            predictions, probabilities = self.predict(df, include_proba=True)
        else:
            predictions = self.predict(df, include_proba=False)
            probabilities = None
        
        # Create results dataframe
        if include_original:
            results = df.copy()
        else:
            results = pd.DataFrame()
        
        # Add predictions
        task_name = self.metadata['model_info']['task_name'] if self.metadata else 'prediction'
        results[f'{task_name}_prediction'] = predictions
        
        # Add probabilities if available
        if probabilities is not None:
            if probabilities.shape[1] == 2:  # Binary classification
                results[f'{task_name}_probability_class_0'] = probabilities[:, 0]
                results[f'{task_name}_probability_class_1'] = probabilities[:, 1]
            else:  # Multiclass
                for i in range(probabilities.shape[1]):
                    results[f'{task_name}_probability_class_{i}'] = probabilities[:, i]
        
        # Save results
        results.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
        print(f"  Total predictions: {len(results)}")
        print(f"\nPrediction distribution:")
        print(results[f'{task_name}_prediction'].value_counts())
        
        return results
    
    def predict_single(
        self,
        features: Dict,
        return_proba: bool = True
    ) -> Dict:
        """
        Make prediction for a single isolate.
        
        Args:
            features: Dictionary with feature names and values
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with prediction and optional probabilities
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        if return_proba:
            prediction, probabilities = self.predict(df, include_proba=True)
            
            result = {
                'prediction': int(prediction[0]),
            }
            
            if probabilities is not None:
                if probabilities.shape[1] == 2:  # Binary
                    result['probability_class_0'] = float(probabilities[0, 0])
                    result['probability_class_1'] = float(probabilities[0, 1])
                else:  # Multiclass
                    for i in range(probabilities.shape[1]):
                        result[f'probability_class_{i}'] = float(probabilities[0, i])
        else:
            prediction = self.predict(df, include_proba=False)
            result = {
                'prediction': int(prediction[0])
            }
        
        return result


def predict_from_csv(
    model_path: str,
    input_csv: str,
    output_csv: str,
    include_proba: bool = True,
    include_original: bool = True
) -> pd.DataFrame:
    """
    Convenience function for batch prediction from CSV.
    
    Args:
        model_path: Path to saved model file
        input_csv: Path to input CSV with new isolates
        output_csv: Path to save predictions
        include_proba: Include prediction probabilities
        include_original: Include original columns in output
        
    Returns:
        DataFrame with predictions
    """
    deployment = ModelDeployment(model_path)
    return deployment.predict_from_csv(
        input_csv,
        output_csv,
        include_proba=include_proba,
        include_original=include_original
    )


def predict_single_isolate(
    model_path: str,
    features: Dict,
    return_proba: bool = True
) -> Dict:
    """
    Convenience function for single isolate prediction.
    
    Args:
        model_path: Path to saved model file
        features: Dictionary with feature values
        return_proba: Whether to return probabilities
        
    Returns:
        Dictionary with prediction and probabilities
    """
    deployment = ModelDeployment(model_path)
    return deployment.predict_single(features, return_proba=return_proba)
