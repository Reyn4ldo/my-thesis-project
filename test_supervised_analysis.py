"""
Unit Tests for Supervised AMR Analysis Module

Tests all three supervised learning tasks and their components.
"""

import unittest
import pandas as pd
import numpy as np
from supervised_analysis import SupervisedAMRAnalysis, quick_supervised_analysis
from data_preparation import AMRDataPreparation


class TestSupervisedAMRAnalysis(unittest.TestCase):
    """Test suite for supervised analysis functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        # Prepare test data
        prep = AMRDataPreparation('rawdata.csv')
        
        # Load and clean data
        df_raw = prep.load_data()
        df_clean = prep.clean_sir_interpretations(df_raw)
        
        # Encode resistance
        df_binary = prep.encode_binary_resistance(df_clean, missing_as_susceptible=True)
        
        # Handle missing values
        df_handled = prep.handle_missing_values(df_binary, strategy='conservative')
        
        # One-hot encode only some context columns, keeping targets
        # For Task 2 and 3, we need to keep bacterial_species, sample_source, administrative_region
        df_encoded = prep.encode_categorical_onehot(
            df_handled, 
            columns=['national_site', 'local_site', 'esbl']
        )
        
        # Scale features
        df_scaled = prep.scale_features(df_encoded, fit=True, exclude_binary=True)
        
        # Drop original int columns
        cols_to_drop = [col for col in prep.antibiotic_int_cols if col in df_scaled.columns]
        cls.df = df_scaled.drop(columns=cols_to_drop)
        
        # Manually create feature groups since we used custom pipeline
        cls.groups = {
            'binary_resistance': [col for col in cls.df.columns if '_binary' in col],
            'ordinal_resistance': [col for col in cls.df.columns if '_ordinal' in col],
            'amr_indices': [col for col in prep.amr_indices if col in cls.df.columns],
            'context_encoded': [col for col in cls.df.columns 
                               if any(ctx in col for ctx in ['national_site_', 'local_site_', 'esbl_'])],
            'metadata': [col for col in cls.df.columns 
                        if col in ['isolate_code', 'replicate', 'colony']]
        }
        
        cls.analyzer = SupervisedAMRAnalysis(cls.df)
    
    def test_01_initialization(self):
        """Test analyzer initialization"""
        analyzer = SupervisedAMRAnalysis(self.df)
        self.assertIsNotNone(analyzer.df)
        self.assertEqual(len(analyzer.df), len(self.df))
        self.assertIsInstance(analyzer.results, dict)
    
    def test_02_create_high_mar_target(self):
        """Test high MAR target creation"""
        # Test default threshold (0.3)
        y = self.analyzer.create_high_mar_target()
        self.assertEqual(len(y), len(self.df))
        self.assertTrue(y.isin([0, 1]).all())
        
        # Test custom threshold
        y_custom = self.analyzer.create_high_mar_target(threshold=0.4)
        self.assertEqual(len(y_custom), len(self.df))
        self.assertTrue(y_custom.sum() < y.sum())  # Higher threshold -> fewer high MAR
    
    def test_03_stratified_split(self):
        """Test stratified data splitting"""
        # Create simple target
        y = self.analyzer.create_high_mar_target()
        X = self.df[self.groups['binary_resistance']]
        
        # Test default split (70/15/15)
        X_train, X_val, X_test, y_train, y_val, y_test = self.analyzer.stratified_split(X, y)
        
        # Check sizes
        total = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total, len(y))
        
        # Check proportions (approximately)
        self.assertAlmostEqual(len(X_train) / total, 0.7, delta=0.05)
        self.assertAlmostEqual(len(X_val) / total, 0.15, delta=0.05)
        self.assertAlmostEqual(len(X_test) / total, 0.15, delta=0.05)
        
        # Check stratification (proportions should be similar)
        train_prop = y_train.mean()
        val_prop = y_val.mean()
        test_prop = y_test.mean()
        overall_prop = y.mean()
        
        self.assertAlmostEqual(train_prop, overall_prop, delta=0.1)
        self.assertAlmostEqual(val_prop, overall_prop, delta=0.1)
        self.assertAlmostEqual(test_prop, overall_prop, delta=0.1)
    
    def test_04_get_algorithm_configs(self):
        """Test algorithm configuration retrieval"""
        configs = self.analyzer.get_algorithm_configs()
        
        # Check all 6 algorithms are present
        expected_algorithms = [
            'LogisticRegression', 'RandomForest', 'GradientBoosting',
            'NaiveBayes', 'SVM', 'KNN'
        ]
        
        for alg in expected_algorithms:
            self.assertIn(alg, configs)
            self.assertIn('model', configs[alg])
            self.assertIn('param_grid', configs[alg])
    
    def test_05_create_pipeline(self):
        """Test pipeline creation"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier()
        numeric_features = self.groups['binary_resistance']
        
        # Test without scaling
        pipeline = self.analyzer.create_pipeline(model, numeric_features, scale=False)
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.steps), 2)  # preprocessor + classifier
        
        # Test with scaling
        pipeline_scaled = self.analyzer.create_pipeline(model, numeric_features, scale=True)
        self.assertIsNotNone(pipeline_scaled)
        self.assertEqual(len(pipeline_scaled.steps), 2)
    
    def test_06_evaluate_model(self):
        """Test model evaluation metrics"""
        # Create dummy predictions
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        # Binary evaluation
        metrics = self.analyzer.evaluate_model(y_true, y_pred, average='binary')
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Check value ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['f1'], 0)
        self.assertLessEqual(metrics['f1'], 1)
    
    def test_07_train_and_evaluate_all_models(self):
        """Test training all 6 models"""
        # Prepare small subset for faster testing
        y = self.analyzer.create_high_mar_target()
        X = self.df[self.groups['binary_resistance'][:10]]  # Use only 10 features
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.analyzer.stratified_split(X, y)
        
        # Train all models (without tuning)
        results = self.analyzer.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='binary'
        )
        
        # Check all 6 models were trained
        self.assertEqual(len(results), 6)
        
        # Check each result has required components
        for name, result in results.items():
            self.assertIn('pipeline', result)
            self.assertIn('val_metrics', result)
            self.assertIn('best_params', result)
            
            # Check metrics
            self.assertIn('accuracy', result['val_metrics'])
            self.assertIn('f1', result['val_metrics'])
    
    def test_08_select_best_model(self):
        """Test best model selection"""
        # Create mock results
        mock_results = {
            'Model1': {'val_metrics': {'f1': 0.8, 'accuracy': 0.75}},
            'Model2': {'val_metrics': {'f1': 0.85, 'accuracy': 0.80}},
            'Model3': {'val_metrics': {'f1': 0.75, 'accuracy': 0.82}}
        }
        
        best_name, best_result = self.analyzer.select_best_model(mock_results, metric='f1')
        
        self.assertEqual(best_name, 'Model2')
        self.assertEqual(best_result['val_metrics']['f1'], 0.85)
    
    def test_09_task1_high_mar_prediction(self):
        """Test Task 1: High MAR prediction"""
        feature_cols = self.groups['binary_resistance'][:15]  # Use subset for speed
        
        results = self.analyzer.task1_high_mar_prediction(
            feature_cols=feature_cols,
            threshold=0.3,
            include_tuning=False  # Skip tuning for faster testing
        )
        
        # Check result structure
        self.assertEqual(results['task'], 'high_mar_prediction')
        self.assertIn('all_models', results)
        self.assertIn('best_model', results)
        self.assertIn('val_metrics', results)
        self.assertIn('test_metrics', results)
        self.assertIn('splits', results)
        
        # Check metrics
        self.assertGreaterEqual(results['test_metrics']['accuracy'], 0)
        self.assertLessEqual(results['test_metrics']['accuracy'], 1)
        self.assertGreaterEqual(results['test_metrics']['f1'], 0)
        self.assertLessEqual(results['test_metrics']['f1'], 1)
    
    def test_10_task2_species_classification(self):
        """Test Task 2: Species classification"""
        feature_cols = self.groups['binary_resistance'][:15]  # Use subset for speed
        
        results = self.analyzer.task2_species_classification(
            feature_cols=feature_cols,
            min_samples=5,  # Lower threshold for testing
            include_tuning=False
        )
        
        # Check result structure
        self.assertEqual(results['task'], 'species_classification')
        self.assertIn('all_models', results)
        self.assertIn('best_model', results)
        self.assertIn('test_metrics', results)
        
        # Check multiclass metrics
        self.assertIn('accuracy', results['test_metrics'])
        self.assertIn('f1', results['test_metrics'])  # Macro F1
        self.assertIn('f1_weighted', results['test_metrics'])
    
    def test_11_task3_region_classification(self):
        """Test Task 3: Region classification"""
        feature_cols = self.groups['binary_resistance'][:15]
        species_cols = [col for col in self.df.columns if 'bacterial_species_' in col]
        feature_cols += species_cols[:5]  # Use subset
        
        results = self.analyzer.task3_region_source_classification(
            feature_cols=feature_cols,
            target_col='administrative_region',
            min_samples=5,
            include_tuning=False
        )
        
        # Check result structure
        self.assertIn('classification', results['task'])
        self.assertIn('all_models', results)
        self.assertIn('best_model', results)
        self.assertIn('test_metrics', results)
        
        # Check metrics
        self.assertIn('accuracy', results['test_metrics'])
        self.assertIn('f1', results['test_metrics'])
        self.assertIn('f1_weighted', results['test_metrics'])
    
    def test_12_task3_source_classification(self):
        """Test Task 3: Source classification"""
        feature_cols = self.groups['binary_resistance'][:15]
        species_cols = [col for col in self.df.columns if 'bacterial_species_' in col]
        feature_cols += species_cols[:5]
        
        results = self.analyzer.task3_region_source_classification(
            feature_cols=feature_cols,
            target_col='sample_source',
            min_samples=5,
            include_tuning=False
        )
        
        # Check result structure
        self.assertIn('classification', results['task'])
        self.assertIn('test_metrics', results)
    
    def test_13_quick_supervised_analysis_single_task(self):
        """Test quick analysis function for single task"""
        results = quick_supervised_analysis(
            self.df,
            task='high_mar',
            include_tuning=False
        )
        
        self.assertIn('high_mar', results)
        self.assertEqual(results['high_mar']['task'], 'high_mar_prediction')
    
    def test_14_quick_supervised_analysis_all_tasks(self):
        """Test quick analysis function for all tasks"""
        results = quick_supervised_analysis(
            self.df,
            task='all',
            include_tuning=False
        )
        
        # Check all tasks are present
        self.assertIn('high_mar', results)
        self.assertIn('species', results)
        self.assertIn('region', results)
        self.assertIn('source', results)
    
    def test_15_confusion_matrix_shape(self):
        """Test confusion matrix shapes"""
        # Binary task
        y_binary = self.analyzer.create_high_mar_target()
        X = self.df[self.groups['binary_resistance'][:10]]
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.analyzer.stratified_split(X, y_binary)
        
        results = self.analyzer.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='binary'
        )
        
        # Check confusion matrix is 2x2 for binary
        for name, result in results.items():
            cm = result['val_metrics']['confusion_matrix']
            self.assertEqual(cm.shape, (2, 2))
    
    def test_16_missing_target_handling(self):
        """Test handling of missing target values"""
        # Create target with some NaN values
        y = self.analyzer.create_high_mar_target()
        y_with_nan = y.copy()
        y_with_nan.iloc[:10] = np.nan
        
        X = self.df[self.groups['binary_resistance'][:10]]
        
        # Split should remove NaN targets
        X_train, X_val, X_test, y_train, y_val, y_test = self.analyzer.stratified_split(
            X, y_with_nan
        )
        
        # Check no NaN in targets
        self.assertFalse(y_train.isna().any())
        self.assertFalse(y_val.isna().any())
        self.assertFalse(y_test.isna().any())
        
        # Check total size is reduced
        total = len(X_train) + len(X_val) + len(X_test)
        self.assertLess(total, len(y_with_nan))
    
    def test_17_rare_class_grouping(self):
        """Test rare class grouping for multiclass tasks"""
        # Task 2 with strict min_samples
        feature_cols = self.groups['binary_resistance'][:10]
        
        results = self.analyzer.task2_species_classification(
            feature_cols=feature_cols,
            min_samples=50,  # High threshold will group many as 'Other'
            include_tuning=False
        )
        
        # Should complete without error
        self.assertIn('test_metrics', results)
    
    def test_18_feature_importance_accessibility(self):
        """Test that we can access feature importance from tree-based models"""
        y = self.analyzer.create_high_mar_target()
        X = self.df[self.groups['binary_resistance'][:10]]
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.analyzer.stratified_split(X, y)
        
        results = self.analyzer.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='binary'
        )
        
        # Check RandomForest has feature_importances_
        rf_pipeline = results['RandomForest']['pipeline']
        clf = rf_pipeline.named_steps['classifier']
        
        self.assertTrue(hasattr(clf, 'feature_importances_'))
    
    def test_19_pipeline_persistence(self):
        """Test that pipelines can be used for prediction"""
        y = self.analyzer.create_high_mar_target()
        X = self.df[self.groups['binary_resistance'][:10]]
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.analyzer.stratified_split(X, y)
        
        results = self.analyzer.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='binary'
        )
        
        # Test prediction with trained pipeline
        for name, result in results.items():
            pipeline = result['pipeline']
            y_pred = pipeline.predict(X_test)
            
            # Check prediction shape
            self.assertEqual(len(y_pred), len(X_test))
            # Check predictions are valid
            self.assertTrue(np.isin(y_pred, [0, 1]).all())
    
    def test_20_custom_feature_set(self):
        """Test with custom feature set"""
        # Use only beta-lactam antibiotics
        beta_lactam_features = [
            col for col in self.groups['binary_resistance']
            if any(ab in col for ab in ['ampicillin', 'cefotaxime', 'ceftiofur'])
        ]
        
        if len(beta_lactam_features) > 0:
            results = self.analyzer.task1_high_mar_prediction(
                feature_cols=beta_lactam_features,
                include_tuning=False
            )
            
            self.assertIn('test_metrics', results)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    print("="*80)
    print("SUPERVISED AMR ANALYSIS - UNIT TESTS")
    print("="*80)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSupervisedAMRAnalysis)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    print("="*80)
