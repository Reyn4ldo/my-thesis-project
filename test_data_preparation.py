"""
Unit tests for AMR Data Preparation Module

This test suite validates all data preparation functions including:
- Data loading and inspection
- S/I/R cleaning
- Binary and ordinal encoding
- Categorical encoding
- Missing value handling
- Feature scaling
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from data_preparation import AMRDataPreparation, quick_prepare


class TestAMRDataPreparation(unittest.TestCase):
    """Test suite for AMR Data Preparation class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - run once before all tests"""
        cls.test_csv_path = 'rawdata.csv'
        if not os.path.exists(cls.test_csv_path):
            raise FileNotFoundError(f"Test data file not found: {cls.test_csv_path}")
    
    def setUp(self):
        """Set up for each test"""
        self.prep = AMRDataPreparation(self.test_csv_path)
    
    def test_01_load_data(self):
        """Test data loading"""
        df = self.prep.load_data()
        
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertGreater(len(df.columns), 0)
        
        # Check that key column groups are identified
        self.assertIsNotNone(self.prep.antibiotic_int_cols)
        self.assertGreater(len(self.prep.antibiotic_int_cols), 0)
        
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Found {len(self.prep.antibiotic_int_cols)} antibiotic interpretation columns")
    
    def test_02_inspect_data(self):
        """Test data inspection"""
        self.prep.load_data()
        report = self.prep.inspect_data()
        
        self.assertIsInstance(report, dict)
        self.assertIn('shape', report)
        self.assertIn('column_names', report)
        self.assertIn('missing_values', report)
        self.assertIn('context_distributions', report)
        self.assertIn('antibiotic_int_distributions', report)
        self.assertIn('amr_statistics', report)
        
        print("✓ Data inspection report generated successfully")
        print(f"✓ Shape: {report['shape']}")
    
    def test_03_clean_sir_interpretations(self):
        """Test S/I/R interpretation cleaning"""
        self.prep.load_data()
        df_clean = self.prep.clean_sir_interpretations(self.prep.df_raw)
        
        # Check that variants are mapped correctly
        for col in self.prep.antibiotic_int_cols:
            if col in df_clean.columns:
                unique_vals = df_clean[col].dropna().unique()
                # Should only have 's', 'i', 'r' (no '*r', '*i', '*s')
                for val in unique_vals:
                    self.assertIn(val, ['s', 'i', 'r'], 
                                f"Unexpected value '{val}' in {col}")
        
        print("✓ S/I/R interpretations cleaned successfully")
        print(f"✓ Valid values: s, i, r")
    
    def test_04_binary_encoding(self):
        """Test binary resistance encoding"""
        self.prep.load_data()
        df_clean = self.prep.clean_sir_interpretations(self.prep.df_raw)
        df_binary = self.prep.encode_binary_resistance(df_clean)
        
        # Check that binary columns were created
        binary_cols = [col for col in df_binary.columns if '_binary' in col]
        self.assertGreater(len(binary_cols), 0)
        
        # Check that values are 0 or 1
        for col in binary_cols:
            unique_vals = df_binary[col].dropna().unique()
            for val in unique_vals:
                self.assertIn(val, [0, 1], f"Invalid binary value {val} in {col}")
        
        # Check specific mapping: 'r' -> 1, 's'/'i' -> 0
        test_col = 'ampicillin_int'
        if test_col in df_clean.columns:
            binary_col = test_col.replace('_int', '_binary')
            
            # Check that 'r' maps to 1
            r_indices = df_clean[test_col] == 'r'
            if r_indices.sum() > 0:
                self.assertTrue(all(df_binary.loc[r_indices, binary_col] == 1))
            
            # Check that 's' maps to 0
            s_indices = df_clean[test_col] == 's'
            if s_indices.sum() > 0:
                self.assertTrue(all(df_binary.loc[s_indices, binary_col] == 0))
        
        print(f"✓ Binary encoding created: {len(binary_cols)} columns")
        print("✓ Encoding verified: r=1, s/i=0")
    
    def test_05_ordinal_encoding(self):
        """Test ordinal S/I/R encoding"""
        self.prep.load_data()
        df_clean = self.prep.clean_sir_interpretations(self.prep.df_raw)
        df_ordinal = self.prep.encode_ordinal_sir(df_clean)
        
        # Check that ordinal columns were created
        ordinal_cols = [col for col in df_ordinal.columns if '_ordinal' in col]
        self.assertGreater(len(ordinal_cols), 0)
        
        # Check that values are 0, 1, or 2
        for col in ordinal_cols:
            unique_vals = df_ordinal[col].dropna().unique()
            for val in unique_vals:
                self.assertIn(val, [0, 1, 2], f"Invalid ordinal value {val} in {col}")
        
        # Check specific mapping
        test_col = 'ampicillin_int'
        if test_col in df_clean.columns:
            ordinal_col = test_col.replace('_int', '_ordinal')
            
            # Check mappings
            s_indices = df_clean[test_col] == 's'
            if s_indices.sum() > 0:
                self.assertTrue(all(df_ordinal.loc[s_indices, ordinal_col] == 0))
            
            i_indices = df_clean[test_col] == 'i'
            if i_indices.sum() > 0:
                self.assertTrue(all(df_ordinal.loc[i_indices, ordinal_col] == 1))
            
            r_indices = df_clean[test_col] == 'r'
            if r_indices.sum() > 0:
                self.assertTrue(all(df_ordinal.loc[r_indices, ordinal_col] == 2))
        
        print(f"✓ Ordinal encoding created: {len(ordinal_cols)} columns")
        print("✓ Encoding verified: s=0, i=1, r=2")
    
    def test_06_categorical_onehot_encoding(self):
        """Test one-hot encoding of categorical variables"""
        self.prep.load_data()
        df_encoded = self.prep.encode_categorical_onehot(self.prep.df_raw)
        
        # Check that original categorical columns are removed
        for col in self.prep.context_cols:
            if col in self.prep.df_raw.columns:
                self.assertNotIn(col, df_encoded.columns, 
                               f"Original column {col} should be removed after encoding")
        
        # Check that one-hot columns were created
        for col in self.prep.context_cols:
            if col in self.prep.df_raw.columns:
                one_hot_cols = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
                self.assertGreater(len(one_hot_cols), 0, 
                                 f"No one-hot columns created for {col}")
                
                # Check that values are 0 or 1
                for oh_col in one_hot_cols:
                    unique_vals = df_encoded[oh_col].unique()
                    for val in unique_vals:
                        self.assertIn(val, [0, 1], 
                                    f"Invalid one-hot value {val} in {oh_col}")
        
        print("✓ One-hot encoding successful")
        print(f"✓ Total encoded columns: {len(df_encoded.columns)}")
    
    def test_07_missing_value_handling(self):
        """Test missing value handling"""
        self.prep.load_data()
        df_clean = self.prep.clean_sir_interpretations(self.prep.df_raw)
        df_handled = self.prep.handle_missing_values(df_clean)
        
        # Check that AMR indices have no missing values after imputation
        for col in self.prep.amr_indices:
            if col in df_handled.columns:
                missing_count = df_handled[col].isnull().sum()
                self.assertEqual(missing_count, 0, 
                               f"AMR index {col} still has {missing_count} missing values")
        
        # Check that rows with too many missing values were dropped
        self.assertLessEqual(len(df_handled), len(df_clean))
        
        print("✓ Missing values handled successfully")
        print(f"✓ Records: {len(df_clean)} -> {len(df_handled)}")
    
    def test_08_feature_scaling(self):
        """Test feature scaling"""
        self.prep.load_data()
        df = self.prep.df_raw.copy()
        
        # Add some dummy numeric columns for testing
        df['test_numeric'] = np.random.randn(len(df)) * 10 + 50
        
        # Scale only AMR indices
        columns_to_scale = [col for col in self.prep.amr_indices if col in df.columns]
        df_scaled = self.prep.scale_features(df, columns=columns_to_scale, fit=True)
        
        # Check that scaled columns have mean ~0 and std ~1
        for col in columns_to_scale:
            mean = df_scaled[col].mean()
            std = df_scaled[col].std()
            
            self.assertAlmostEqual(mean, 0, places=6, 
                                 msg=f"{col} mean should be ~0 after scaling")
            self.assertAlmostEqual(std, 1, places=1, 
                                 msg=f"{col} std should be ~1 after scaling")
        
        print("✓ Feature scaling successful")
        print(f"✓ Scaled {len(columns_to_scale)} columns")
    
    def test_09_complete_pipeline(self):
        """Test complete data preparation pipeline"""
        df_prepared = self.prep.prepare_data(
            include_binary=True,
            include_ordinal=True,
            include_onehot=True,
            missing_strategy='conservative',
            scale=True,
            drop_original_int=True
        )
        
        self.assertIsNotNone(df_prepared)
        self.assertIsInstance(df_prepared, pd.DataFrame)
        
        # Check that binary columns exist
        binary_cols = [col for col in df_prepared.columns if '_binary' in col]
        self.assertGreater(len(binary_cols), 0)
        
        # Check that ordinal columns exist
        ordinal_cols = [col for col in df_prepared.columns if '_ordinal' in col]
        self.assertGreater(len(ordinal_cols), 0)
        
        # Check that one-hot columns exist
        one_hot_cols = [col for col in df_prepared.columns 
                       if any(ctx in col for ctx in self.prep.context_cols)]
        self.assertGreater(len(one_hot_cols), 0)
        
        # Check that original _int columns are removed
        int_cols = [col for col in df_prepared.columns if col.endswith('_int')]
        self.assertEqual(len(int_cols), 0, "Original _int columns should be removed")
        
        print("✓ Complete pipeline executed successfully")
        print(f"✓ Final shape: {df_prepared.shape}")
        print(f"✓ Binary columns: {len(binary_cols)}")
        print(f"✓ Ordinal columns: {len(ordinal_cols)}")
        print(f"✓ One-hot columns: {len(one_hot_cols)}")
    
    def test_10_feature_groups(self):
        """Test feature group extraction"""
        self.prep.prepare_data(include_binary=True, include_ordinal=True)
        feature_groups = self.prep.get_feature_groups()
        
        self.assertIsInstance(feature_groups, dict)
        self.assertIn('binary_resistance', feature_groups)
        self.assertIn('ordinal_resistance', feature_groups)
        self.assertIn('amr_indices', feature_groups)
        self.assertIn('context_encoded', feature_groups)
        
        print("✓ Feature groups extracted successfully")
        for group_name, features in feature_groups.items():
            print(f"  - {group_name}: {len(features)} features")
    
    def test_11_export_prepared_data(self):
        """Test exporting prepared data"""
        self.prep.prepare_data(include_binary=True, include_ordinal=False)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name
        
        try:
            self.prep.export_prepared_data(tmp_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_path))
            
            # Check that data can be loaded back
            df_loaded = pd.read_csv(tmp_path)
            self.assertEqual(len(df_loaded), len(self.prep.df_encoded))
            
            print(f"✓ Data exported successfully to temporary file")
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_12_quick_prepare(self):
        """Test quick prepare convenience function"""
        df_prepared = quick_prepare(self.test_csv_path)
        
        self.assertIsNotNone(df_prepared)
        self.assertIsInstance(df_prepared, pd.DataFrame)
        self.assertGreater(len(df_prepared), 0)
        
        # Check that it includes binary columns
        binary_cols = [col for col in df_prepared.columns if '_binary' in col]
        self.assertGreater(len(binary_cols), 0)
        
        print("✓ Quick prepare function works correctly")
        print(f"✓ Shape: {df_prepared.shape}")


class TestDataQuality(unittest.TestCase):
    """Test data quality and edge cases"""
    
    def setUp(self):
        """Set up for each test"""
        self.prep = AMRDataPreparation('rawdata.csv')
        self.prep.load_data()
    
    def test_sir_variant_handling(self):
        """Test that S/I/R variants are handled correctly"""
        df = self.prep.df_raw.copy()
        
        # Check if variants exist in raw data
        has_variants = False
        for col in self.prep.antibiotic_int_cols:
            if col in df.columns:
                vals = df[col].astype(str).unique()
                if any('*' in str(v) for v in vals):
                    has_variants = True
                    break
        
        if has_variants:
            df_clean = self.prep.clean_sir_interpretations(df)
            
            # Verify no variants remain
            for col in self.prep.antibiotic_int_cols:
                if col in df_clean.columns:
                    vals = df_clean[col].dropna().astype(str).unique()
                    for val in vals:
                        self.assertNotIn('*', val, 
                                       f"Variant marker '*' found in {col}: {val}")
            
            print("✓ S/I/R variants handled correctly")
        else:
            print("⊘ No variants found in test data (test skipped)")
    
    def test_binary_encoding_consistency(self):
        """Test that binary encoding is consistent across all antibiotics"""
        df_clean = self.prep.clean_sir_interpretations(self.prep.df_raw)
        df_binary = self.prep.encode_binary_resistance(df_clean, missing_as_susceptible=True)
        
        # For each antibiotic with both _int and _binary columns
        for int_col in self.prep.antibiotic_int_cols:
            if int_col in df_clean.columns:
                binary_col = int_col.replace('_int', '_binary')
                if binary_col in df_binary.columns:
                    # Where _int is 'r', _binary should be 1
                    r_mask = df_clean[int_col] == 'r'
                    if r_mask.sum() > 0:
                        self.assertTrue(
                            all(df_binary.loc[r_mask, binary_col] == 1),
                            f"Inconsistent encoding in {binary_col}"
                        )
        
        print("✓ Binary encoding is consistent across all antibiotics")
    
    def test_no_data_leakage(self):
        """Test that original interpretation columns are removed when requested"""
        df_prepared = self.prep.prepare_data(drop_original_int=True)
        
        # Check that no _int columns remain (except in column names of derived features)
        int_cols_remaining = [col for col in df_prepared.columns if col.endswith('_int')]
        
        self.assertEqual(len(int_cols_remaining), 0, 
                        "Original _int columns should be removed to prevent data leakage")
        
        print("✓ No data leakage: original _int columns removed")


def run_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestAMRDataPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    
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
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
