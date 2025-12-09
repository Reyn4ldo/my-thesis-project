"""
Phase 1: Data Preparation Module for AMR Analysis

This module implements comprehensive data preparation for antimicrobial resistance (AMR) analysis,
including data ingestion, cleaning, encoding, and transformation.

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings


class AMRDataPreparation:
    """
    Comprehensive data preparation class for Antimicrobial Resistance (AMR) data analysis.
    
    This class handles:
    - Data ingestion and inspection
    - S/I/R interpretation cleaning
    - Binary and ordinal resistance encoding
    - Categorical feature encoding (one-hot)
    - Missing value imputation
    - Feature scaling
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the AMR data preparation pipeline.
        
        Args:
            filepath (str): Path to the CSV file containing raw data
        """
        self.filepath = filepath
        self.df_raw = None
        self.df_cleaned = None
        self.df_encoded = None
        
        # Define key variable groups
        self.antibiotic_int_cols = None
        self.antibiotic_mic_cols = None
        self.context_cols = [
            'bacterial_species',
            'sample_source', 
            'administrative_region',
            'national_site',
            'local_site',
            'esbl'
        ]
        self.amr_indices = ['mar_index', 'scored_resistance', 'num_antibiotics_tested']
        
        # Transformation artifacts
        self.scaler = None
        self.one_hot_encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data and perform initial inspection.
        
        Returns:
            pd.DataFrame: Raw dataframe loaded from CSV
        """
        self.df_raw = pd.read_csv(self.filepath)
        
        # Identify antibiotic columns
        self.antibiotic_int_cols = [col for col in self.df_raw.columns if col.endswith('_int')]
        self.antibiotic_mic_cols = [col for col in self.df_raw.columns if col.endswith('_mic')]
        
        return self.df_raw
    
    def inspect_data(self) -> Dict:
        """
        Inspect data for column names, types, missing values, and distributions.
        
        Returns:
            Dict: Dictionary containing inspection results
        """
        if self.df_raw is None:
            self.load_data()
        
        inspection_report = {
            'shape': self.df_raw.shape,
            'column_names': list(self.df_raw.columns),
            'column_types': self.df_raw.dtypes.to_dict(),
            'missing_values': self.df_raw.isnull().sum().to_dict(),
            'missing_percentages': (self.df_raw.isnull().sum() / len(self.df_raw) * 100).to_dict(),
        }
        
        # Context variable distributions
        inspection_report['context_distributions'] = {}
        for col in self.context_cols:
            if col in self.df_raw.columns:
                inspection_report['context_distributions'][col] = {
                    'unique_count': self.df_raw[col].nunique(),
                    'value_counts': self.df_raw[col].value_counts().to_dict()
                }
        
        # Antibiotic interpretation distributions
        inspection_report['antibiotic_int_distributions'] = {}
        for col in self.antibiotic_int_cols[:5]:  # Sample first 5
            inspection_report['antibiotic_int_distributions'][col] = {
                'unique_values': self.df_raw[col].value_counts().to_dict(),
                'missing_count': self.df_raw[col].isnull().sum()
            }
        
        # AMR indices statistics
        inspection_report['amr_statistics'] = {}
        for col in self.amr_indices:
            if col in self.df_raw.columns:
                inspection_report['amr_statistics'][col] = {
                    'mean': float(self.df_raw[col].mean()),
                    'std': float(self.df_raw[col].std()),
                    'min': float(self.df_raw[col].min()),
                    'max': float(self.df_raw[col].max()),
                    'missing': int(self.df_raw[col].isnull().sum())
                }
        
        return inspection_report
    
    def clean_sir_interpretations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean S/I/R interpretation columns.
        
        Steps:
        1. Convert to lowercase
        2. Map variants (*r -> r, *i -> i, *s -> s)
        3. Treat blank or unknown as missing
        
        Args:
            df (pd.DataFrame): Dataframe with raw interpretations
            
        Returns:
            pd.DataFrame: Dataframe with cleaned interpretations
        """
        df_clean = df.copy()
        
        for col in self.antibiotic_int_cols:
            if col in df_clean.columns:
                # Convert to lowercase
                df_clean[col] = df_clean[col].astype(str).str.lower()
                
                # Map variants: *r -> r, *i -> i, *s -> s
                df_clean[col] = df_clean[col].replace({
                    '*r': 'r',
                    '*i': 'i', 
                    '*s': 's',
                    '': np.nan,
                    'unknown': np.nan
                })
                
                # Ensure only valid values remain
                valid_values = ['s', 'i', 'r']
                df_clean.loc[~df_clean[col].isin(valid_values), col] = np.nan
        
        return df_clean
    
    def encode_binary_resistance(self, df: pd.DataFrame, 
                                  missing_as_susceptible: bool = True) -> pd.DataFrame:
        """
        Create binary resistance encoding for each antibiotic.
        
        Encoding:
        - R = 1 (resistant)
        - S = 0 (susceptible)
        - I = 0 (intermediate, treated as susceptible)
        - Missing = 0 (if missing_as_susceptible=True) or NaN
        
        Args:
            df (pd.DataFrame): Dataframe with cleaned interpretations
            missing_as_susceptible (bool): If True, treat missing as susceptible (0)
            
        Returns:
            pd.DataFrame: Dataframe with additional binary resistance columns
        """
        df_binary = df.copy()
        
        for col in self.antibiotic_int_cols:
            if col in df_binary.columns:
                # Create binary column name
                binary_col = col.replace('_int', '_binary')
                
                # Encode: r=1, s/i=0
                df_binary[binary_col] = df_binary[col].map({
                    'r': 1,
                    's': 0,
                    'i': 0
                })
                
                # Handle missing values
                if missing_as_susceptible:
                    df_binary[binary_col] = df_binary[binary_col].fillna(0)
        
        return df_binary
    
    def encode_ordinal_sir(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ordinal S/I/R encoding for each antibiotic.
        
        Encoding:
        - S = 0 (susceptible)
        - I = 1 (intermediate)
        - R = 2 (resistant)
        - Missing = NaN
        
        Args:
            df (pd.DataFrame): Dataframe with cleaned interpretations
            
        Returns:
            pd.DataFrame: Dataframe with additional ordinal S/I/R columns
        """
        df_ordinal = df.copy()
        
        for col in self.antibiotic_int_cols:
            if col in df_ordinal.columns:
                # Create ordinal column name
                ordinal_col = col.replace('_int', '_ordinal')
                
                # Encode: s=0, i=1, r=2
                df_ordinal[ordinal_col] = df_ordinal[col].map({
                    's': 0,
                    'i': 1,
                    'r': 2
                })
        
        return df_ordinal
    
    def encode_categorical_onehot(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   drop_first: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical context variables.
        
        Args:
            df (pd.DataFrame): Dataframe with categorical columns
            columns (List[str], optional): Specific columns to encode. If None, uses all context_cols
            drop_first (bool): If True, drop first category to avoid multicollinearity
            
        Returns:
            pd.DataFrame: Dataframe with one-hot encoded columns
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = [col for col in self.context_cols if col in df.columns]
        
        for col in columns:
            if col in df_encoded.columns:
                # Create one-hot encoded columns
                one_hot = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first, dtype=int)
                
                # Drop original column and add one-hot columns
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, one_hot], axis=1)
                
                # Store encoder info for later use
                self.one_hot_encoders[col] = list(one_hot.columns)
        
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'conservative',
                             drop_threshold: float = 0.8) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategies:
        - 'conservative': Treat missing antibiotic interpretations as susceptible
        - 'median': Impute numeric fields with median
        - 'drop': Drop rows with too many missing values
        
        Args:
            df (pd.DataFrame): Dataframe with missing values
            strategy (str): Strategy for handling missing values
            drop_threshold (float): Drop rows if missing % exceeds this threshold
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_clean = df.copy()
        
        # Drop rows with too many missing values (based on key columns only)
        # Consider only antibiotic interpretation and context columns for missingness assessment
        key_cols = self.antibiotic_int_cols + [col for col in self.context_cols if col in df_clean.columns]
        key_cols_present = [col for col in key_cols if col in df_clean.columns]
        
        if len(key_cols_present) > 0:
            missing_per_row = df_clean[key_cols_present].isnull().sum(axis=1) / len(key_cols_present)
        else:
            missing_per_row = df_clean.isnull().sum(axis=1) / len(df_clean.columns)
        
        rows_to_drop = missing_per_row > drop_threshold
        if rows_to_drop.sum() > 0:
            warnings.warn(f"Dropping {rows_to_drop.sum()} rows with >{drop_threshold*100}% missing values")
            df_clean = df_clean[~rows_to_drop]
        
        # Handle AMR indices missing values with median
        for col in self.amr_indices:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        return df_clean
    
    def scale_features(self, df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      fit: bool = True,
                      exclude_binary: bool = True) -> pd.DataFrame:
        """
        Standardize numeric features using z-score normalization.
        
        Formula: X_scaled = (X - mean) / std
        
        Args:
            df (pd.DataFrame): Dataframe with numeric columns
            columns (List[str], optional): Columns to scale. If None, scales numeric columns intelligently
            fit (bool): If True, fit the scaler. If False, use existing fitted scaler
            exclude_binary (bool): If True, exclude binary columns from scaling
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            # Default: scale AMR indices, MIC values, and optionally ordinal columns
            columns = []
            for col in df_scaled.columns:
                if df_scaled[col].dtype in ['float64', 'int64']:
                    # Include AMR indices (using class attribute for consistency)
                    if col in self.amr_indices:
                        columns.append(col)
                    # Include MIC columns
                    elif '_mic' in col:
                        columns.append(col)
                    # Include ordinal (but not binary if exclude_binary is True)
                    elif '_ordinal' in col:
                        columns.append(col)
                    elif '_binary' in col and not exclude_binary:
                        columns.append(col)
        
        if len(columns) > 0:
            if fit:
                self.scaler = StandardScaler()
                df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Set fit=True first.")
                df_scaled[columns] = self.scaler.transform(df_scaled[columns])
        
        return df_scaled
    
    def prepare_data(self,
                    include_binary: bool = True,
                    include_ordinal: bool = False,
                    include_onehot: bool = True,
                    missing_strategy: str = 'conservative',
                    scale: bool = True,
                    drop_original_int: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        This method orchestrates all data preparation steps:
        1. Load data
        2. Clean S/I/R interpretations
        3. Encode resistance (binary and/or ordinal)
        4. Encode categorical variables (one-hot)
        5. Handle missing values
        6. Scale features
        
        Args:
            include_binary (bool): Include binary resistance encoding
            include_ordinal (bool): Include ordinal S/I/R encoding
            include_onehot (bool): Include one-hot encoding for categorical variables
            missing_strategy (str): Strategy for handling missing values
            scale (bool): Apply feature scaling
            drop_original_int (bool): Drop original *_int columns after encoding
            
        Returns:
            pd.DataFrame: Fully prepared dataframe ready for analysis
        """
        # Step 1: Load data
        if self.df_raw is None:
            self.load_data()
        
        df = self.df_raw.copy()
        
        # Step 2: Clean S/I/R interpretations
        print("Cleaning S/I/R interpretations...")
        df = self.clean_sir_interpretations(df)
        
        # Step 3: Encode resistance
        if include_binary:
            print("Creating binary resistance encoding...")
            df = self.encode_binary_resistance(df, 
                                               missing_as_susceptible=(missing_strategy == 'conservative'))
        
        if include_ordinal:
            print("Creating ordinal S/I/R encoding...")
            df = self.encode_ordinal_sir(df)
        
        # Step 4: Handle missing values
        print("Handling missing values...")
        df = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Step 5: One-hot encode categorical variables
        if include_onehot:
            print("One-hot encoding categorical variables...")
            df = self.encode_categorical_onehot(df)
        
        # Step 6: Scale features
        if scale:
            print("Scaling numeric features...")
            df = self.scale_features(df, fit=True)
        
        # Step 7: Drop original interpretation columns if requested
        if drop_original_int:
            cols_to_drop = [col for col in self.antibiotic_int_cols if col in df.columns]
            df = df.drop(columns=cols_to_drop)
        
        self.df_encoded = df
        print(f"Data preparation complete! Final shape: {df.shape}")
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get organized groups of features from the prepared dataset.
        
        Returns:
            Dict: Dictionary with feature groups
        """
        if self.df_encoded is None:
            raise ValueError("Data not prepared yet. Call prepare_data() first.")
        
        feature_groups = {
            'binary_resistance': [col for col in self.df_encoded.columns if '_binary' in col],
            'ordinal_resistance': [col for col in self.df_encoded.columns if '_ordinal' in col],
            'amr_indices': [col for col in self.amr_indices if col in self.df_encoded.columns],
            'context_encoded': [col for col in self.df_encoded.columns 
                               if any(ctx in col for ctx in self.context_cols)],
            'metadata': [col for col in self.df_encoded.columns 
                        if col in ['isolate_code', 'replicate', 'colony']]
        }
        
        return feature_groups
    
    def export_prepared_data(self, output_path: str):
        """
        Export prepared data to CSV file.
        
        Args:
            output_path (str): Path to save the prepared data
        """
        if self.df_encoded is None:
            raise ValueError("Data not prepared yet. Call prepare_data() first.")
        
        self.df_encoded.to_csv(output_path, index=False)
        print(f"Prepared data exported to: {output_path}")


def quick_prepare(filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Quick data preparation with default settings.
    
    This convenience function provides a simple interface for standard data preparation.
    
    Args:
        filepath (str): Path to raw CSV file
        output_path (str, optional): Path to save prepared data
        
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    prep = AMRDataPreparation(filepath)
    df_prepared = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        missing_strategy='conservative',
        scale=True,
        drop_original_int=True
    )
    
    if output_path:
        prep.export_prepared_data(output_path)
    
    return df_prepared


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("AMR Data Preparation Module - Example Usage")
    print("="*80)
    
    # Initialize data preparation
    prep = AMRDataPreparation('rawdata.csv')
    
    # Load and inspect data
    print("\n1. Loading data...")
    prep.load_data()
    print(f"   Loaded {prep.df_raw.shape[0]} records with {prep.df_raw.shape[1]} columns")
    
    print("\n2. Inspecting data...")
    inspection = prep.inspect_data()
    print(f"   Found {len(prep.antibiotic_int_cols)} antibiotic interpretation columns")
    print(f"   Found {len(prep.context_cols)} context columns")
    
    print("\n3. Preparing data with default pipeline...")
    df_prepared = prep.prepare_data(
        include_binary=True,
        include_ordinal=True,
        include_onehot=True,
        missing_strategy='conservative',
        scale=True,
        drop_original_int=True
    )
    
    print("\n4. Feature groups:")
    feature_groups = prep.get_feature_groups()
    for group_name, features in feature_groups.items():
        print(f"   {group_name}: {len(features)} features")
    
    print("\n5. Exporting prepared data...")
    prep.export_prepared_data('prepared_data.csv')
    
    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)
