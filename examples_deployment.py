"""
Example Usage for Model Deployment

This script demonstrates how to:
1. Train and save a high MAR prediction model
2. Load and use the saved model for predictions
3. Deploy the model on new data

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis
from model_deployment import ModelDeployment, predict_from_csv


def example_1_train_and_save_model():
    """
    Example 1: Train a high MAR prediction model and save it for deployment.
    
    This creates a complete deployment package including:
    - Trained pipeline (high_MAR_model.pkl)
    - Model metadata (high_MAR_model_metadata.json)
    """
    print("="*80)
    print("EXAMPLE 1: TRAIN AND SAVE MODEL")
    print("="*80)
    
    # Step 1: Prepare data
    print("\n1. Preparing data...")
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,
        drop_original_int=True
    )
    
    # Step 2: Get features
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    print(f"   Using {len(feature_cols)} binary resistance features")
    
    # Step 3: Train model with auto-save
    print("\n2. Training model...")
    analyzer = SupervisedAMRAnalysis(df)
    results = analyzer.task1_high_mar_prediction(
        feature_cols=feature_cols,
        threshold=0.3,
        include_tuning=True,
        tune_top_n=3,
        save_model_path='high_MAR_model.pkl'  # Auto-save after training
    )
    
    print(f"\n3. Model saved!")
    print(f"   Best model: {results['best_model']}")
    print(f"   Test F1 score: {results['test_metrics']['f1']:.4f}")
    print(f"   Files created:")
    print(f"     - high_MAR_model.pkl")
    print(f"     - high_MAR_model_metadata.json")
    
    return results


def example_2_load_and_predict():
    """
    Example 2: Load a saved model and make predictions on new data.
    
    This demonstrates the deployment workflow.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: LOAD MODEL AND PREDICT")
    print("="*80)
    
    # Step 1: Load the saved model
    print("\n1. Loading saved model...")
    deployment = ModelDeployment('high_MAR_model.pkl')
    
    # Step 2: View model information
    print("\n2. Model information:")
    info = deployment.get_model_info()
    print(f"   Task: {info['task_name']}")
    print(f"   Model type: {info['model_type']}")
    print(f"   Created: {info['created_at']}")
    
    metrics = deployment.get_performance_metrics()
    print(f"\n   Test performance:")
    print(f"     Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"     F1 Score: {metrics['test']['f1']:.4f}")
    
    # Step 3: Get required features
    features = deployment.get_required_features()
    print(f"\n3. Required features: {len(features)}")
    print(f"   Sample features: {features[:3]}")
    
    # Step 4: Make predictions on new data
    # For this example, we'll prepare some test data
    print("\n4. Preparing new data for prediction...")
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,
        drop_original_int=True
    )
    
    # Take a subset as "new" data
    new_data = df[features].sample(n=50, random_state=42)
    new_data.to_csv('new_isolates.csv', index=False)
    
    # Step 5: Batch prediction from CSV
    print("\n5. Making batch predictions...")
    results = deployment.predict_from_csv(
        input_csv='new_isolates.csv',
        output_csv='predictions.csv',
        include_proba=True,
        include_original=True
    )
    
    print(f"\n6. Predictions complete!")
    print(f"   Output saved to: predictions.csv")
    print(f"   Total predictions: {len(results)}")
    print(f"\n   Prediction summary:")
    print(results['high_mar_prediction_prediction'].value_counts())


def example_3_single_isolate_prediction():
    """
    Example 3: Make prediction for a single isolate (e.g., for web app).
    
    This is useful for real-time prediction in interactive applications.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: SINGLE ISOLATE PREDICTION")
    print("="*80)
    
    # Load model
    print("\n1. Loading model...")
    deployment = ModelDeployment('high_MAR_model.pkl')
    
    # Prepare a single isolate's data
    print("\n2. Preparing single isolate data...")
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=True,
        scale=False,
        drop_original_int=True
    )
    
    # Get required features
    features = deployment.get_required_features()
    
    # Get one sample
    sample_features = df[features].iloc[0].to_dict()
    
    # Make prediction
    print("\n3. Making prediction...")
    result = deployment.predict_single(sample_features, return_proba=True)
    
    print(f"\n4. Prediction result:")
    print(f"   Prediction: {result['prediction']} ({'High MAR' if result['prediction'] == 1 else 'Low MAR'})")
    print(f"   Probability Low MAR: {result['probability_class_0']:.4f}")
    print(f"   Probability High MAR: {result['probability_class_1']:.4f}")


def example_4_command_line_deployment():
    """
    Example 4: Using the command-line deployment script.
    
    This demonstrates how to use deploy_model.py for production deployment.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: COMMAND-LINE DEPLOYMENT")
    print("="*80)
    
    print("\n1. View model information:")
    print("   python deploy_model.py --model high_MAR_model.pkl --info")
    
    print("\n2. Make predictions from CSV:")
    print("   python deploy_model.py \\")
    print("       --model high_MAR_model.pkl \\")
    print("       --input new_isolates.csv \\")
    print("       --output predictions.csv")
    
    print("\n3. Predictions without probabilities:")
    print("   python deploy_model.py \\")
    print("       --model high_MAR_model.pkl \\")
    print("       --input new_isolates.csv \\")
    print("       --output predictions.csv \\")
    print("       --no-proba")
    
    print("\n4. Only prediction columns (no original data):")
    print("   python deploy_model.py \\")
    print("       --model high_MAR_model.pkl \\")
    print("       --input new_isolates.csv \\")
    print("       --output predictions.csv \\")
    print("       --no-original")


def main():
    """Run all examples"""
    import sys
    
    print("\n" + "="*80)
    print("MODEL DEPLOYMENT EXAMPLES")
    print("="*80)
    
    # Check if model already exists
    import os
    if not os.path.exists('high_MAR_model.pkl'):
        print("\nNo saved model found. Training a new model...")
        example_1_train_and_save_model()
    else:
        print("\nFound existing model. Skipping training.")
        print("(Delete high_MAR_model.pkl to train a new model)")
    
    # Run prediction examples
    try:
        example_2_load_and_predict()
        example_3_single_isolate_prediction()
        example_4_command_line_deployment()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
