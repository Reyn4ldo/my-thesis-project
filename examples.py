"""
Example Usage Scripts for AMR Data Preparation

This script demonstrates various use cases for the data preparation module.
"""

import pandas as pd
from data_preparation import AMRDataPreparation, quick_prepare


def example_1_quick_start():
    """
    Example 1: Quick Start - Simplest way to prepare data
    """
    print("="*80)
    print("EXAMPLE 1: Quick Start")
    print("="*80)
    
    # One-line data preparation with defaults
    df = quick_prepare('rawdata.csv', output_path='prepared_data_quick.csv')
    
    print(f"\n✓ Prepared data shape: {df.shape}")
    print(f"✓ Output saved to: prepared_data_quick.csv")
    
    return df


def example_2_supervised_learning_setup():
    """
    Example 2: Prepare data for supervised learning (classification/regression)
    
    Use case: Predict MAR index or classify high/low resistance
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Supervised Learning Setup")
    print("="*80)
    
    prep = AMRDataPreparation('rawdata.csv')
    
    # Prepare with binary encoding (best for supervised tasks)
    df = prep.prepare_data(
        include_binary=True,      # Binary resistance (R=1, S/I=0)
        include_ordinal=False,    # Don't need ordinal for classification
        include_onehot=True,      # Include context as features
        missing_strategy='conservative',
        scale=True,
        drop_original_int=True
    )
    
    # Get feature groups
    groups = prep.get_feature_groups()
    
    # Prepare X (features) and y (target)
    # Option A: Predict MAR index
    X_features = (groups['binary_resistance'] + 
                  groups['context_encoded'])
    y_target = 'mar_index'
    
    print(f"\n✓ Feature columns: {len(X_features)}")
    print(f"✓ Target variable: {y_target}")
    print(f"\nSample features:")
    for i, feat in enumerate(X_features[:5], 1):
        print(f"  {i}. {feat}")
    
    # Create datasets
    X = df[X_features]
    y = df[y_target]
    
    print(f"\n✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    
    return X, y, df


def example_3_unsupervised_clustering():
    """
    Example 3: Prepare data for unsupervised learning (clustering, PCA)
    
    Use case: Discover resistance patterns, cluster similar isolates
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Unsupervised Learning - Clustering")
    print("="*80)
    
    prep = AMRDataPreparation('rawdata.csv')
    
    # Prepare with ordinal encoding (preserves resistance gradation)
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,     # S=0, I=1, R=2 preserves order
        include_onehot=True,
        scale=True               # Essential for distance-based clustering
    )
    
    groups = prep.get_feature_groups()
    
    # Features for clustering
    clustering_features = groups['ordinal_resistance'] + groups['amr_indices']
    
    print(f"\n✓ Clustering features: {len(clustering_features)}")
    print(f"\nFeature categories:")
    print(f"  - Ordinal resistance: {len(groups['ordinal_resistance'])} features")
    print(f"  - AMR indices: {len(groups['amr_indices'])} features")
    
    X_clustering = df[clustering_features].dropna()
    
    print(f"\n✓ Data ready for clustering: {X_clustering.shape}")
    print(f"✓ Suggested algorithms: K-means, DBSCAN, Hierarchical")
    
    return X_clustering, df


def example_4_specific_antibiotic_classes():
    """
    Example 4: Focus on specific antibiotic classes
    
    Use case: Analyze beta-lactam resistance specifically
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Specific Antibiotic Classes")
    print("="*80)
    
    prep = AMRDataPreparation('rawdata.csv')
    prep.load_data()
    
    # Clean and encode
    df = prep.clean_sir_interpretations(prep.df_raw)
    df = prep.encode_binary_resistance(df)
    
    # Define antibiotic classes
    beta_lactams = [
        'ampicillin_binary',
        'amoxicillin/clavulanic_acid_binary',
        'cefotaxime_binary',
        'ceftiofur_binary',
        'cefalotin_binary',
        'cefalexin_binary'
    ]
    
    fluoroquinolones = [
        'nalidixic_acid_binary',
        'enrofloxacin_binary',
        'marbofloxacin_binary'
    ]
    
    tetracyclines = [
        'tetracycline_binary',
        'doxycycline_binary'
    ]
    
    print("\n✓ Beta-lactams:")
    for ab in beta_lactams:
        if ab in df.columns:
            resistance_rate = df[ab].mean() * 100
            print(f"  - {ab}: {resistance_rate:.1f}% resistant")
    
    print("\n✓ Fluoroquinolones:")
    for ab in fluoroquinolones:
        if ab in df.columns:
            resistance_rate = df[ab].mean() * 100
            print(f"  - {ab}: {resistance_rate:.1f}% resistant")
    
    # Create dataset with specific classes
    selected_antibiotics = [ab for ab in beta_lactams + fluoroquinolones if ab in df.columns]
    X_specific = df[selected_antibiotics]
    
    print(f"\n✓ Selected {len(selected_antibiotics)} antibiotics from 2 classes")
    
    return X_specific, df


def example_5_by_context():
    """
    Example 5: Analyze data by context (species, source, region)
    
    Use case: Compare resistance patterns across different contexts
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Analysis by Context")
    print("="*80)
    
    prep = AMRDataPreparation('rawdata.csv')
    df_raw = prep.load_data()
    
    # Clean and encode but keep original context columns
    df = prep.clean_sir_interpretations(df_raw)
    df = prep.encode_binary_resistance(df)
    
    # Keep context columns for grouping
    context_cols = ['bacterial_species', 'sample_source', 'administrative_region']
    
    print("\n✓ Resistance by Bacterial Species:")
    species_resistance = df.groupby('bacterial_species')['ampicillin_binary'].agg(['mean', 'count'])
    species_resistance = species_resistance.sort_values('mean', ascending=False)
    print(species_resistance.head())
    
    print("\n✓ Resistance by Sample Source:")
    source_resistance = df.groupby('sample_source')['ampicillin_binary'].agg(['mean', 'count'])
    source_resistance = source_resistance.sort_values('mean', ascending=False)
    print(source_resistance.head())
    
    print("\n✓ Resistance by Region:")
    region_resistance = df.groupby('administrative_region')['mar_index'].mean().sort_values(ascending=False)
    print(region_resistance)
    
    return df


def example_6_custom_pipeline():
    """
    Example 6: Custom pipeline with manual control
    
    Use case: Fine-grained control over each step
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Pipeline")
    print("="*80)
    
    prep = AMRDataPreparation('rawdata.csv')
    
    # Step 1: Load
    print("\nStep 1: Loading data...")
    df = prep.load_data()
    print(f"  Loaded: {df.shape}")
    
    # Step 2: Clean
    print("\nStep 2: Cleaning S/I/R interpretations...")
    df = prep.clean_sir_interpretations(df)
    print(f"  Cleaned: {df.shape}")
    
    # Step 3: Binary encoding only
    print("\nStep 3: Binary encoding...")
    df = prep.encode_binary_resistance(df, missing_as_susceptible=True)
    binary_cols = [col for col in df.columns if '_binary' in col]
    print(f"  Created {len(binary_cols)} binary columns")
    
    # Step 4: Handle missing in AMR indices only
    print("\nStep 4: Imputing AMR indices...")
    for col in prep.amr_indices:
        if col in df.columns:
            before = df[col].isnull().sum()
            df[col] = df[col].fillna(df[col].median())
            after = df[col].isnull().sum()
            print(f"  {col}: {before} → {after} missing")
    
    # Step 5: One-hot encode only specific columns
    print("\nStep 5: Encoding bacterial species only...")
    df = prep.encode_categorical_onehot(df, columns=['bacterial_species'])
    species_cols = [col for col in df.columns if 'bacterial_species_' in col]
    print(f"  Created {len(species_cols)} species columns")
    
    # Step 6: Scale only AMR indices
    print("\nStep 6: Scaling AMR indices...")
    df = prep.scale_features(df, columns=prep.amr_indices, fit=True)
    print(f"  Scaled {len(prep.amr_indices)} columns")
    
    print(f"\n✓ Final shape: {df.shape}")
    
    return df


def example_7_export_multiple_versions():
    """
    Example 7: Create and export multiple versions of prepared data
    
    Use case: Different preparations for different analyses
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Export Multiple Versions")
    print("="*80)
    
    # Version 1: For supervised learning
    print("\nVersion 1: Supervised learning (binary encoding)...")
    prep1 = AMRDataPreparation('rawdata.csv')
    df1 = prep1.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=True
    )
    prep1.export_prepared_data('prepared_supervised.csv')
    print(f"  ✓ Saved to: prepared_supervised.csv ({df1.shape})")
    
    # Version 2: For unsupervised learning
    print("\nVersion 2: Unsupervised learning (ordinal encoding)...")
    prep2 = AMRDataPreparation('rawdata.csv')
    df2 = prep2.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=True,
        scale=True
    )
    prep2.export_prepared_data('prepared_unsupervised.csv')
    print(f"  ✓ Saved to: prepared_unsupervised.csv ({df2.shape})")
    
    # Version 3: Both encodings for flexibility
    print("\nVersion 3: Both encodings (maximum flexibility)...")
    prep3 = AMRDataPreparation('rawdata.csv')
    df3 = prep3.prepare_data(
        include_binary=True,
        include_ordinal=True,
        include_onehot=True,
        scale=True
    )
    prep3.export_prepared_data('prepared_complete.csv')
    print(f"  ✓ Saved to: prepared_complete.csv ({df3.shape})")
    
    print("\n✓ All versions exported successfully!")
    
    return df1, df2, df3


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("AMR DATA PREPARATION - USAGE EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        df_quick = example_1_quick_start()
        X, y, df_supervised = example_2_supervised_learning_setup()
        X_cluster, df_unsupervised = example_3_unsupervised_clustering()
        X_specific, df_classes = example_4_specific_antibiotic_classes()
        df_context = example_5_by_context()
        df_custom = example_6_custom_pipeline()
        df1, df2, df3 = example_7_export_multiple_versions()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - prepared_data_quick.csv")
        print("  - prepared_supervised.csv")
        print("  - prepared_unsupervised.csv")
        print("  - prepared_complete.csv")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
