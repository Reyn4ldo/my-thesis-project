"""
Unit Tests for Model Deployment Module

Tests model saving, loading, and deployment functionality.
"""

import unittest
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from supervised_analysis import SupervisedAMRAnalysis
from model_deployment import ModelDeployment, predict_from_csv, predict_single_isolate
from data_preparation import AMRDataPreparation


class TestModelDeployment(unittest.TestCase):
    """Test suite for model deployment functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and train a simple model once for all tests"""
        # Prepare test data
        prep = AMRDataPreparation('rawdata.csv')
        df = prep.prepare_data(
            include_binary=True,
            include_ordinal=False,
            include_onehot=True,
            scale=False,
            drop_original_int=True
        )
        
        # Get feature groups
        groups = prep.get_feature_groups()
        cls.feature_cols = groups['binary_resistance'][:10]  # Use only 10 features for quick test
        
        cls.df = df
        cls.test_model_path = '/tmp/test_high_MAR_model.pkl'
        cls.test_csv_input = '/tmp/test_input.csv'
        cls.test_csv_output = '/tmp/test_predictions.csv'
        
        # Train a simple model for testing
        print("\nTraining test model...")
        analyzer = SupervisedAMRAnalysis(df)
        
        # Create target
        y = analyzer.create_high_mar_target(threshold=0.3)
        X = df[cls.feature_cols]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = analyzer.stratified_split(X, y)
        
        # Train just one model (RandomForest) for testing
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Train on train+val
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        pipeline.fit(X_train_full, y_train_full)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
        
        # Save model with metadata
        analyzer.save_pipeline_with_metadata(
            pipeline=pipeline,
            filepath=cls.test_model_path,
            task_name='high_mar_prediction',
            feature_cols=cls.feature_cols,
            model_type='RandomForest',
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            train_metrics=test_metrics,
            val_metrics=test_metrics,
            test_metrics=test_metrics,
            splits={'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test)},
            additional_info={'threshold': 0.3}
        )
        
        # Create test CSV file
        test_data = X_test.head(20).copy()
        test_data.to_csv(cls.test_csv_input, index=False)
        
        cls.X_test = X_test
        cls.y_test = y_test
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        for filepath in [cls.test_model_path, 
                        cls.test_model_path.replace('.pkl', '_metadata.json'),
                        cls.test_csv_input,
                        cls.test_csv_output]:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_01_model_saved(self):
        """Test that model file was created"""
        self.assertTrue(Path(self.test_model_path).exists())
        self.assertTrue(Path(self.test_model_path.replace('.pkl', '_metadata.json')).exists())
    
    def test_02_metadata_content(self):
        """Test metadata file contains expected information"""
        metadata_path = self.test_model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check required keys
        self.assertIn('model_info', metadata)
        self.assertIn('features', metadata)
        self.assertIn('metrics', metadata)
        self.assertIn('data_splits', metadata)
        
        # Check model info
        self.assertEqual(metadata['model_info']['task_name'], 'high_mar_prediction')
        self.assertEqual(metadata['model_info']['model_type'], 'RandomForest')
        
        # Check features
        self.assertEqual(len(metadata['features']['feature_columns']), len(self.feature_cols))
        self.assertEqual(metadata['features']['num_features'], len(self.feature_cols))
    
    def test_03_load_model(self):
        """Test loading saved model"""
        deployment = ModelDeployment(self.test_model_path)
        
        self.assertIsNotNone(deployment.pipeline)
        self.assertIsNotNone(deployment.metadata)
        self.assertEqual(deployment.metadata['model_info']['task_name'], 'high_mar_prediction')
    
    def test_04_get_required_features(self):
        """Test getting required features"""
        deployment = ModelDeployment(self.test_model_path)
        features = deployment.get_required_features()
        
        self.assertEqual(len(features), len(self.feature_cols))
        self.assertEqual(set(features), set(self.feature_cols))
    
    def test_05_predict_with_proba(self):
        """Test making predictions with probabilities"""
        deployment = ModelDeployment(self.test_model_path)
        
        # Use test data
        X_sample = self.X_test.head(10)
        predictions, probabilities = deployment.predict(X_sample, include_proba=True)
        
        # Check shapes
        self.assertEqual(len(predictions), 10)
        self.assertEqual(probabilities.shape, (10, 2))  # Binary classification
        
        # Check predictions are 0 or 1
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(10))
    
    def test_06_predict_without_proba(self):
        """Test making predictions without probabilities"""
        deployment = ModelDeployment(self.test_model_path)
        
        X_sample = self.X_test.head(10)
        predictions = deployment.predict(X_sample, include_proba=False)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
    
    def test_07_predict_from_csv(self):
        """Test batch prediction from CSV"""
        deployment = ModelDeployment(self.test_model_path)
        
        results = deployment.predict_from_csv(
            input_csv=self.test_csv_input,
            output_csv=self.test_csv_output,
            include_proba=True,
            include_original=True
        )
        
        # Check output file exists
        self.assertTrue(Path(self.test_csv_output).exists())
        
        # Check results
        self.assertEqual(len(results), 20)  # We saved 20 samples
        self.assertIn('high_mar_prediction_prediction', results.columns)
        self.assertIn('high_mar_prediction_probability_class_0', results.columns)
        self.assertIn('high_mar_prediction_probability_class_1', results.columns)
        
        # Check all original columns are included
        for col in self.feature_cols:
            self.assertIn(col, results.columns)
    
    def test_08_predict_single_isolate(self):
        """Test single isolate prediction"""
        deployment = ModelDeployment(self.test_model_path)
        
        # Get single sample features
        features = self.X_test.iloc[0].to_dict()
        
        result = deployment.predict_single(features, return_proba=True)
        
        # Check result structure
        self.assertIn('prediction', result)
        self.assertIn('probability_class_0', result)
        self.assertIn('probability_class_1', result)
        
        # Check values
        self.assertIn(result['prediction'], [0, 1])
        self.assertAlmostEqual(
            result['probability_class_0'] + result['probability_class_1'],
            1.0,
            places=5
        )
    
    def test_09_convenience_function_csv(self):
        """Test convenience function for CSV prediction"""
        results = predict_from_csv(
            model_path=self.test_model_path,
            input_csv=self.test_csv_input,
            output_csv=self.test_csv_output,
            include_proba=True,
            include_original=False
        )
        
        # Check only prediction columns are included
        self.assertEqual(len(results), 20)
        self.assertIn('high_mar_prediction_prediction', results.columns)
        
        # Original feature columns should not be included
        for col in self.feature_cols:
            self.assertNotIn(col, results.columns)
    
    def test_10_convenience_function_single(self):
        """Test convenience function for single prediction"""
        features = self.X_test.iloc[0].to_dict()
        
        result = predict_single_isolate(
            model_path=self.test_model_path,
            features=features,
            return_proba=True
        )
        
        self.assertIn('prediction', result)
        self.assertIn('probability_class_0', result)
    
    def test_11_missing_features_error(self):
        """Test error handling for missing features"""
        deployment = ModelDeployment(self.test_model_path)
        
        # Create data with missing features
        X_incomplete = self.X_test[self.feature_cols[:5]]  # Only half the features
        
        with self.assertRaises(ValueError) as context:
            deployment.predict(X_incomplete)
        
        self.assertIn("Missing required features", str(context.exception))
    
    def test_12_get_model_info(self):
        """Test getting model information"""
        deployment = ModelDeployment(self.test_model_path)
        
        info = deployment.get_model_info()
        
        self.assertIn('task_name', info)
        self.assertIn('model_type', info)
        self.assertIn('hyperparameters', info)
        self.assertEqual(info['task_name'], 'high_mar_prediction')
    
    def test_13_get_performance_metrics(self):
        """Test getting performance metrics"""
        deployment = ModelDeployment(self.test_model_path)
        
        metrics = deployment.get_performance_metrics()
        
        self.assertIn('test', metrics)
        self.assertIn('validation', metrics)
        self.assertIn('accuracy', metrics['test'])
        self.assertIn('f1', metrics['test'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
