"""
Example Usage Scripts for Supervised AMR Analysis

This script demonstrates the three supervised learning tasks:
1. Binary classification: Predict high MAR/MDR
2. Multiclass classification: Predict bacterial species
3. Multiclass classification: Predict region/source
"""

import pandas as pd
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis, quick_supervised_analysis


def example_1_task1_high_mar_prediction():
    """
    Example 1: Task 1 - Binary classification to predict high MAR/MDR
    
    Uses binary resistance features to predict whether an isolate has
    high multiple antibiotic resistance (MAR >= 0.3).
    """
    print("="*80)
    print("EXAMPLE 1: TASK 1 - HIGH MAR/MDR PREDICTION")
    print("="*80)
    
    # Prepare data with binary encoding
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,  # Scaling handled in pipeline
        drop_original_int=True
    )
    
    # Get feature groups
    groups = prep.get_feature_groups()
    
    # Features: binary resistance + optional context
    feature_cols = groups['binary_resistance']
    # Optionally add context features:
    # feature_cols += groups['context_encoded']
    
    print(f"\nFeatures: {len(feature_cols)} binary resistance columns")
    print(f"Sample features: {feature_cols[:5]}")
    
    # Run Task 1
    analyzer = SupervisedAMRAnalysis(df)
    results = analyzer.task1_high_mar_prediction(
        feature_cols=feature_cols,
        threshold=0.3,
        include_tuning=True,
        tune_top_n=3
    )
    
    print("\n" + "="*80)
    print("TASK 1 RESULTS SUMMARY")
    print("="*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"\nValidation Metrics:")
    for metric, value in results['val_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    print(f"\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    return results


def example_2_task2_species_classification():
    """
    Example 2: Task 2 - Multiclass classification to predict bacterial species
    
    Uses only AMR resistance patterns to predict bacterial species.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: TASK 2 - SPECIES CLASSIFICATION")
    print("="*80)
    
    # Prepare data with binary encoding
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=False,  # Don't include species in features!
        scale=False,
        drop_original_int=True
    )
    
    # Features: only AMR patterns
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    
    print(f"\nFeatures: {len(feature_cols)} AMR resistance columns")
    
    # Run Task 2
    analyzer = SupervisedAMRAnalysis(df)
    results = analyzer.task2_species_classification(
        feature_cols=feature_cols,
        min_samples=10,  # Group species with <10 samples as 'Other'
        include_tuning=True,
        tune_top_n=3
    )
    
    print("\n" + "="*80)
    print("TASK 2 RESULTS SUMMARY")
    print("="*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"\nValidation Metrics:")
    for metric, value in results['val_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    print(f"\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    return results


def example_3_task3_region_classification():
    """
    Example 3: Task 3 - Multiclass classification to predict administrative region
    
    Uses AMR patterns + species to predict region.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: TASK 3 - REGION CLASSIFICATION")
    print("="*80)
    
    # Prepare data with binary encoding and species one-hot
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,  # Include species and other context
        scale=False,
        drop_original_int=True
    )
    
    # Features: AMR + species (but not region!)
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    
    # Add species one-hot columns
    species_cols = [col for col in df.columns if 'bacterial_species_' in col]
    feature_cols += species_cols
    
    print(f"\nFeatures: {len(feature_cols)} total")
    print(f"  - AMR features: {len(groups['binary_resistance'])}")
    print(f"  - Species features: {len(species_cols)}")
    
    # Run Task 3 for region
    analyzer = SupervisedAMRAnalysis(df)
    results = analyzer.task3_region_source_classification(
        feature_cols=feature_cols,
        target_col='administrative_region',
        min_samples=10,
        include_tuning=True,
        tune_top_n=3
    )
    
    print("\n" + "="*80)
    print("TASK 3 RESULTS SUMMARY (REGION)")
    print("="*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"\nValidation Metrics:")
    for metric, value in results['val_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    print(f"\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    return results


def example_4_task3_source_classification():
    """
    Example 4: Task 3 - Multiclass classification to predict sample source
    
    Uses AMR patterns + species to predict source.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: TASK 3 - SOURCE CLASSIFICATION")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,
        drop_original_int=True
    )
    
    # Features: AMR + species
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    species_cols = [col for col in df.columns if 'bacterial_species_' in col]
    feature_cols += species_cols
    
    print(f"\nFeatures: {len(feature_cols)} total")
    
    # Run Task 3 for source
    analyzer = SupervisedAMRAnalysis(df)
    results = analyzer.task3_region_source_classification(
        feature_cols=feature_cols,
        target_col='sample_source',
        min_samples=10,
        include_tuning=True,
        tune_top_n=3
    )
    
    print("\n" + "="*80)
    print("TASK 3 RESULTS SUMMARY (SOURCE)")
    print("="*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"\nValidation Metrics:")
    for metric, value in results['val_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    print(f"\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    return results


def example_5_quick_all_tasks():
    """
    Example 5: Quick function to run all tasks at once
    
    Uses the convenience function for rapid analysis.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: QUICK ALL TASKS")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,
        drop_original_int=True
    )
    
    # Run all tasks with quick function
    results = quick_supervised_analysis(
        df,
        task='all',
        include_tuning=False,  # Skip tuning for speed
        tune_top_n=0
    )
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED")
    print("="*80)
    
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper()}:")
        print(f"  Best Model: {task_results['best_model']}")
        print(f"  Test F1: {task_results['test_metrics']['f1']:.4f}")
        print(f"  Test Accuracy: {task_results['test_metrics']['accuracy']:.4f}")
    
    return results


def example_6_compare_feature_sets():
    """
    Example 6: Compare different feature sets for Task 1
    
    Tests whether adding context improves high MAR prediction.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: COMPARE FEATURE SETS FOR HIGH MAR PREDICTION")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,
        drop_original_int=True
    )
    
    groups = prep.get_feature_groups()
    analyzer = SupervisedAMRAnalysis(df)
    
    # Test 1: Only AMR features
    print("\nTest 1: Only AMR resistance features")
    print("-" * 40)
    feature_cols_amr = groups['binary_resistance']
    results_amr = analyzer.task1_high_mar_prediction(
        feature_cols=feature_cols_amr,
        include_tuning=False
    )
    
    # Test 2: AMR + species
    print("\nTest 2: AMR + species features")
    print("-" * 40)
    species_cols = [col for col in df.columns if 'bacterial_species_' in col]
    feature_cols_species = groups['binary_resistance'] + species_cols
    
    analyzer2 = SupervisedAMRAnalysis(df)
    results_species = analyzer2.task1_high_mar_prediction(
        feature_cols=feature_cols_species,
        include_tuning=False
    )
    
    # Test 3: AMR + species + source
    print("\nTest 3: AMR + species + source features")
    print("-" * 40)
    source_cols = [col for col in df.columns if 'sample_source_' in col]
    feature_cols_full = groups['binary_resistance'] + species_cols + source_cols
    
    analyzer3 = SupervisedAMRAnalysis(df)
    results_full = analyzer3.task1_high_mar_prediction(
        feature_cols=feature_cols_full,
        include_tuning=False
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Feature Set': ['AMR only', 'AMR + Species', 'AMR + Species + Source'],
        'Num Features': [
            len(feature_cols_amr),
            len(feature_cols_species),
            len(feature_cols_full)
        ],
        'Best Model': [
            results_amr['best_model'],
            results_species['best_model'],
            results_full['best_model']
        ],
        'Test F1': [
            results_amr['test_metrics']['f1'],
            results_species['test_metrics']['f1'],
            results_full['test_metrics']['f1']
        ],
        'Test Accuracy': [
            results_amr['test_metrics']['accuracy'],
            results_species['test_metrics']['accuracy'],
            results_full['test_metrics']['accuracy']
        ]
    })
    
    print(comparison.to_string(index=False))
    
    return {
        'amr_only': results_amr,
        'amr_species': results_species,
        'amr_species_source': results_full,
        'comparison': comparison
    }


def example_7_custom_threshold():
    """
    Example 7: Test different MAR thresholds for Task 1
    
    Explores how threshold selection affects model performance.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: CUSTOM MAR THRESHOLDS")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=False,
        scale=False,
        drop_original_int=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    
    # Test different thresholds
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    threshold_results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold}")
        print("-" * 40)
        
        analyzer = SupervisedAMRAnalysis(df)
        results = analyzer.task1_high_mar_prediction(
            feature_cols=feature_cols,
            threshold=threshold,
            include_tuning=False
        )
        
        threshold_results.append({
            'threshold': threshold,
            'best_model': results['best_model'],
            'test_f1': results['test_metrics']['f1'],
            'test_accuracy': results['test_metrics']['accuracy']
        })
    
    # Summary
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame(threshold_results)
    print(comparison_df.to_string(index=False))
    
    return threshold_results


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("SUPERVISED AMR ANALYSIS - USAGE EXAMPLES")
    print("="*80)
    
    try:
        # Run examples
        results_1 = example_1_task1_high_mar_prediction()
        results_2 = example_2_task2_species_classification()
        results_3 = example_3_task3_region_classification()
        results_4 = example_4_task3_source_classification()
        results_5 = example_5_quick_all_tasks()
        results_6 = example_6_compare_feature_sets()
        results_7 = example_7_custom_threshold()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'task1': results_1,
            'task2': results_2,
            'task3_region': results_3,
            'task3_source': results_4,
            'quick_all': results_5,
            'feature_comparison': results_6,
            'threshold_comparison': results_7
        }
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
