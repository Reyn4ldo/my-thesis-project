"""
Complete End-to-End Workflow: From Raw Data to Deployed Model

This script demonstrates the complete AMR analysis pipeline:
Phase 1: Data Preparation
Phase 3: Model Training
Phase 4: Model Deployment

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis
from model_deployment import ModelDeployment


def main():
    """
    Complete workflow demonstrating all phases.
    """
    print("\n" + "="*80)
    print("COMPLETE END-TO-END AMR ANALYSIS WORKFLOW")
    print("="*80)
    
    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA PREPARATION")
    print("="*80)
    
    print("\nStep 1.1: Load and inspect raw data")
    prep = AMRDataPreparation('rawdata.csv')
    df_raw = prep.load_data()
    
    inspection = prep.inspect_data()
    print(f"  Total samples: {inspection['shape'][0]}")
    print(f"  Total columns: {inspection['shape'][1]}")
    print(f"  Column types: {len(inspection['column_types'])}")
    
    print("\nStep 1.2: Prepare data for supervised learning")
    df_prepared = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,  # Scaling handled in pipeline
        drop_original_int=True
    )
    
    print(f"  Final shape: {df_prepared.shape}")
    
    print("\nStep 1.3: Identify feature groups")
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    print(f"  Binary resistance features: {len(feature_cols)}")
    print(f"  Context features: {len(groups.get('context_encoded', []))}")
    print(f"  AMR indices: {len(groups.get('amr_indices', []))}")
    
    # =========================================================================
    # PHASE 3: SUPERVISED LEARNING
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: SUPERVISED LEARNING")
    print("="*80)
    
    print("\nStep 3.1: Initialize analyzer")
    analyzer = SupervisedAMRAnalysis(df_prepared)
    
    print("\nStep 3.2: Train and evaluate models")
    print("  Task: Predict high MAR/MDR")
    print("  Threshold: 0.3 (30% resistance)")
    print("  Features: Binary resistance patterns")
    
    # Train with auto-save
    results = analyzer.task1_high_mar_prediction(
        feature_cols=feature_cols,
        threshold=0.3,
        include_tuning=True,
        tune_top_n=3,
        save_model_path='high_MAR_production_model.pkl'
    )
    
    print(f"\nStep 3.3: Model evaluation complete")
    print(f"  Best model: {results['best_model']}")
    print(f"  Validation F1: {results['val_metrics']['f1']:.4f}")
    print(f"  Test F1: {results['test_metrics']['f1']:.4f}")
    print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    
    # =========================================================================
    # PHASE 4: MODEL DEPLOYMENT
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: MODEL DEPLOYMENT")
    print("="*80)
    
    print("\nStep 4.1: Load production model")
    deployment = ModelDeployment('high_MAR_production_model.pkl')
    
    print("\nStep 4.2: Review model metadata")
    info = deployment.get_model_info()
    print(f"  Task: {info['task_name']}")
    print(f"  Model: {info['model_type']}")
    print(f"  Created: {info['created_at']}")
    
    metrics = deployment.get_performance_metrics()
    print(f"\n  Performance:")
    print(f"    Accuracy: {metrics['test']['accuracy']:.2%}")
    print(f"    Precision: {metrics['test']['precision']:.2%}")
    print(f"    Recall: {metrics['test']['recall']:.2%}")
    print(f"    F1 Score: {metrics['test']['f1']:.2%}")
    
    print("\nStep 4.3: Simulate new isolates for prediction")
    # Take a sample from test set to simulate new data
    required_features = deployment.get_required_features()
    new_isolates = df_prepared[required_features].sample(n=30, random_state=99)
    new_isolates.to_csv('production_input.csv', index=False)
    print(f"  Created 30 'new' isolates in production_input.csv")
    
    print("\nStep 4.4: Make predictions")
    predictions = deployment.predict_from_csv(
        input_csv='production_input.csv',
        output_csv='production_predictions.csv',
        include_proba=True,
        include_original=True
    )
    
    print("\nStep 4.5: Analyze prediction results")
    high_mar_count = predictions['high_mar_prediction_prediction'].sum()
    low_mar_count = len(predictions) - high_mar_count
    
    print(f"  Predictions summary:")
    print(f"    Low MAR: {low_mar_count} isolates ({low_mar_count/len(predictions)*100:.1f}%)")
    print(f"    High MAR: {high_mar_count} isolates ({high_mar_count/len(predictions)*100:.1f}%)")
    
    # Show some example predictions with high confidence
    print("\n  Sample high-confidence predictions:")
    predictions['max_prob'] = predictions[['high_mar_prediction_probability_class_0', 
                                           'high_mar_prediction_probability_class_1']].max(axis=1)
    top_confident = predictions.nlargest(3, 'max_prob')
    
    for idx, row in top_confident.iterrows():
        prediction = 'High MAR' if row['high_mar_prediction_prediction'] == 1 else 'Low MAR'
        confidence = row['max_prob']
        print(f"    Isolate {idx}: {prediction} (confidence: {confidence:.2%})")
    
    # =========================================================================
    # DEPLOYMENT SCENARIOS
    # =========================================================================
    print("\n" + "="*80)
    print("DEPLOYMENT SCENARIOS")
    print("="*80)
    
    print("\nScenario 1: Batch Processing")
    print("  Use case: Process daily lab results")
    print("  Command:")
    print("    python deploy_model.py \\")
    print("      --model high_MAR_production_model.pkl \\")
    print("      --input daily_results.csv \\")
    print("      --output daily_predictions.csv")
    
    print("\nScenario 2: Single Isolate (Interactive)")
    print("  Use case: Real-time decision support")
    
    # Get a single sample
    sample_features = new_isolates.iloc[0].to_dict()
    result = deployment.predict_single(sample_features, return_proba=True)
    
    prediction = 'High MAR' if result['prediction'] == 1 else 'Low MAR'
    confidence = result[f'probability_class_{result["prediction"]}']
    
    print(f"  Sample prediction: {prediction}")
    print(f"  Confidence: {confidence:.2%}")
    print("\n  Code:")
    print("    features = {...}  # Antibiotic resistance pattern")
    print("    result = deployment.predict_single(features)")
    print("    print(f'Prediction: {result[\"prediction\"]}')")
    
    print("\nScenario 3: Surveillance System")
    print("  Use case: Monitor trends over time")
    print("  Features:")
    print("    - Aggregate predictions by region/site/time")
    print("    - Track MDR rates and trends")
    print("    - Generate alerts for concerning patterns")
    print("    - Visualize geographic distribution")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    
    print("\nFiles created:")
    print("  1. high_MAR_production_model.pkl - Trained model")
    print("  2. high_MAR_production_model_metadata.json - Model documentation")
    print("  3. production_input.csv - Sample new isolates")
    print("  4. production_predictions.csv - Predictions with probabilities")
    
    print("\nNext steps:")
    print("  1. Review predictions in production_predictions.csv")
    print("  2. Integrate model into production systems")
    print("  3. Monitor model performance over time")
    print("  4. Retrain periodically with new data")
    
    print("\n" + "="*80)
    print("END-TO-END WORKFLOW SUCCESSFUL")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
