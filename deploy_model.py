#!/usr/bin/env python
"""
Deploy AMR Model - Command-line Script

This script demonstrates how to use saved AMR models for deployment.
It can be used as a standalone command-line tool or as a template for
custom deployment solutions.

Usage:
    python deploy_model.py --model high_MAR_model.pkl --input new_data.csv --output predictions.csv

Author: Thesis Project
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from model_deployment import ModelDeployment


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(
        description='Deploy trained AMR model for predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python deploy_model.py --model high_MAR_model.pkl --input new_data.csv --output predictions.csv
  
  # Without probabilities
  python deploy_model.py --model high_MAR_model.pkl --input new_data.csv --output predictions.csv --no-proba
  
  # Only predictions (no original columns)
  python deploy_model.py --model high_MAR_model.pkl --input new_data.csv --output predictions.csv --no-original
  
  # Show model info
  python deploy_model.py --model high_MAR_model.pkl --info
        """
    )
    
    parser.add_argument(
        '--model',
        required=True,
        help='Path to saved model file (.pkl)'
    )
    
    parser.add_argument(
        '--input',
        help='Path to input CSV file with new isolates'
    )
    
    parser.add_argument(
        '--output',
        help='Path to save predictions CSV'
    )
    
    parser.add_argument(
        '--no-proba',
        action='store_true',
        help='Do not include prediction probabilities'
    )
    
    parser.add_argument(
        '--no-original',
        action='store_true',
        help='Do not include original columns in output'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display model information and exit'
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    print("="*80)
    print("AMR MODEL DEPLOYMENT")
    print("="*80)
    deployment = ModelDeployment(args.model)
    
    # Display model info
    if args.info:
        print("\n" + "="*80)
        print("MODEL INFORMATION")
        print("="*80)
        
        info = deployment.get_model_info()
        print(f"\nModel Details:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\nRequired Features: {len(deployment.get_required_features())}")
        print(f"  Sample features: {deployment.get_required_features()[:5]}")
        
        metrics = deployment.get_performance_metrics()
        if 'test' in metrics and metrics['test']:
            print(f"\nTest Performance:")
            for metric, value in metrics['test'].items():
                if metric != 'confusion_matrix':
                    print(f"  {metric}: {value:.4f}")
        
        print("="*80)
        sys.exit(0)
    
    # Check if input/output provided
    if not args.input or not args.output:
        print("\nError: --input and --output are required for prediction")
        print("Use --info to display model information only")
        sys.exit(1)
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Make predictions
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)
    
    try:
        results = deployment.predict_from_csv(
            input_csv=args.input,
            output_csv=args.output,
            include_proba=not args.no_proba,
            include_original=not args.no_original
        )
        
        print("\n" + "="*80)
        print("DEPLOYMENT SUCCESSFUL")
        print("="*80)
        print(f"Predictions saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
