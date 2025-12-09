"""
Phase 3: Supervised Pattern Recognition for AMR Analysis

This module implements supervised machine learning pipelines for three classification tasks:
1. Binary classification: Predict high MAR/MDR
2. Multiclass classification: Predict bacterial species from resistance profile
3. Multiclass classification: Predict region/source from resistance + species

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class SupervisedAMRAnalysis:
    """
    Supervised learning analysis for AMR data.
    
    Implements three classification tasks:
    1. Binary: Predict high MAR/MDR
    2. Multiclass: Predict bacterial species
    3. Multiclass: Predict region or source
    
    Common pipeline: 70/15/15 stratified split + 6 algorithms with hyperparameter tuning
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize supervised analysis.
        
        Args:
            df: Prepared dataframe from data_preparation module
        """
        self.df = df.copy()
        self.results = {}
        
    def create_high_mar_target(self, threshold: float = 0.3) -> pd.Series:
        """
        Create binary target for high MAR classification.
        
        Args:
            threshold: MAR index threshold for high resistance (default: 0.3)
            
        Returns:
            pd.Series: Binary target (1 = high MAR, 0 = low MAR)
        """
        if 'mar_index' not in self.df.columns:
            raise ValueError("mar_index column not found in dataframe")
        
        high_mar = (self.df['mar_index'] >= threshold).astype(int)
        return high_mar
    
    def stratified_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Perform stratified 70/15/15 split.
        
        Args:
            X: Features
            y: Target variable
            train_size: Training set proportion (default: 0.7)
            val_size: Validation set proportion (default: 0.15)
            test_size: Test set proportion (default: 0.15)
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Remove rows with missing target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Verify proportions sum to 1
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Proportions must sum to 1.0, got {total}")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_size + test_size),
            stratify=y,
            random_state=random_state
        )
        
        # Second split: val vs test
        val_proportion = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_proportion),
            stratify=y_temp,
            random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_algorithm_configs(self) -> Dict:
        """
        Get configuration for all 6 algorithms.
        
        Returns:
            Dict: Algorithm configurations with base models and hyperparameter grids
        """
        configs = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'param_grid': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l2'],
                    'classifier__solver': ['lbfgs']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
            },
            'NaiveBayes': {
                'model': GaussianNB(),
                'param_grid': {
                    'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'param_grid': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf'],
                    'classifier__gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'param_grid': {
                    'classifier__n_neighbors': [3, 5, 7, 9, 11],
                    'classifier__weights': ['uniform', 'distance']
                }
            }
        }
        
        return configs
    
    def create_pipeline(
        self,
        model,
        numeric_features: List[str],
        scale: bool = True
    ) -> Pipeline:
        """
        Create preprocessing + model pipeline.
        
        Args:
            model: Scikit-learn classifier
            numeric_features: List of numeric feature names
            scale: Whether to scale features (for SVM, kNN)
            
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        steps = []
        
        # Preprocessing
        if scale:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numeric_features)
                ],
                remainder='passthrough'
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', SimpleImputer(strategy='median'), numeric_features)
                ],
                remainder='passthrough'
            )
        
        steps.append(('preprocessor', preprocessor))
        steps.append(('classifier', model))
        
        return Pipeline(steps)
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'binary',
        labels: Optional[List] = None
    ) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method ('binary', 'macro', 'weighted')
            labels: List of labels for multiclass
            
        Returns:
            Dict: Evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels)
        }
        
        return metrics
    
    def train_and_evaluate_all_models(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        task_type: str = 'binary',
        scale_features: Optional[List[str]] = None
    ) -> Dict:
        """
        Train and evaluate all 6 algorithms.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target
            task_type: 'binary' or 'multiclass'
            scale_features: List of features to scale (defaults to all numeric)
            
        Returns:
            Dict: Results for all models
        """
        results = {}
        configs = self.get_algorithm_configs()
        
        # Determine features to scale
        if scale_features is None:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            scale_features = numeric_cols
        
        # Set average method for metrics
        average = 'binary' if task_type == 'binary' else 'macro'
        
        for name, config in configs.items():
            print(f"\nTraining {name}...")
            
            # Determine if scaling is needed
            needs_scaling = name in ['SVM', 'KNN', 'LogisticRegression']
            
            # Create pipeline
            pipeline = self.create_pipeline(
                config['model'],
                scale_features,
                scale=needs_scaling
            )
            
            # Train baseline model
            pipeline.fit(X_train, y_train)
            
            # Predict on validation
            y_pred = pipeline.predict(X_val)
            
            # Evaluate
            metrics = self.evaluate_model(y_val, y_pred, average=average)
            
            results[name] = {
                'pipeline': pipeline,
                'val_metrics': metrics,
                'best_params': None
            }
            
            print(f"  Validation F1: {metrics['f1']:.4f}")
            print(f"  Validation Accuracy: {metrics['accuracy']:.4f}")
        
        return results
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        param_grid: Dict,
        pipeline: Pipeline,
        cv: int = 5,
        scoring: str = 'f1'
    ) -> Tuple[Pipeline, Dict]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            param_grid: Hyperparameter grid
            pipeline: Base pipeline
            cv: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Tuple of (best pipeline, best parameters)
        """
        print(f"\nTuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def select_best_model(self, results: Dict, metric: str = 'f1') -> Tuple[str, Dict]:
        """
        Select best model based on validation metrics.
        
        Args:
            results: Dictionary of model results
            metric: Metric to use for selection (default: 'f1')
            
        Returns:
            Tuple of (best model name, best model results)
        """
        best_name = None
        best_score = -1
        
        for name, result in results.items():
            score = result['val_metrics'][metric]
            if score > best_score:
                best_score = score
                best_name = name
        
        print(f"\nBest model: {best_name} (Validation {metric.upper()}: {best_score:.4f})")
        
        return best_name, results[best_name]
    
    def final_evaluation(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        task_type: str = 'binary'
    ) -> Dict:
        """
        Final training on train+val and evaluation on test.
        
        Args:
            pipeline: Trained pipeline
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Target sets
            task_type: 'binary' or 'multiclass'
            
        Returns:
            Dict: Final test metrics
        """
        print("\nFinal evaluation...")
        
        # Combine train + val
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        
        # Retrain on train_full
        pipeline.fit(X_train_full, y_train_full)
        
        # Predict on test
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        average = 'binary' if task_type == 'binary' else 'macro'
        metrics = self.evaluate_model(y_test, y_pred, average=average)
        
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test Precision: {metrics['precision']:.4f}")
        print(f"  Test Recall: {metrics['recall']:.4f}")
        print(f"  Test F1: {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        return metrics
    
    def task1_high_mar_prediction(
        self,
        feature_cols: List[str],
        threshold: float = 0.3,
        include_tuning: bool = True,
        tune_top_n: int = 3
    ) -> Dict:
        """
        Task 1: Binary classification to predict high MAR/MDR.
        
        Args:
            feature_cols: List of feature column names
            threshold: MAR index threshold (default: 0.3)
            include_tuning: Whether to perform hyperparameter tuning
            tune_top_n: Number of top models to tune
            
        Returns:
            Dict: Complete results including all models and final test metrics
        """
        print("="*80)
        print("TASK 1: HIGH MAR/MDR PREDICTION")
        print("="*80)
        
        # Create target
        y = self.create_high_mar_target(threshold)
        X = self.df[feature_cols]
        
        print(f"\nTarget distribution:")
        print(f"  High MAR (1): {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Low MAR (0): {(y == 0).sum()} ({(1-y.mean())*100:.1f}%)")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.stratified_split(X, y)
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Train and evaluate all models
        results = self.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='binary'
        )
        
        # Hyperparameter tuning for top models
        if include_tuning:
            # Sort by validation F1
            sorted_models = sorted(
                results.items(),
                key=lambda x: x[1]['val_metrics']['f1'],
                reverse=True
            )
            
            configs = self.get_algorithm_configs()
            
            for name, result in sorted_models[:tune_top_n]:
                tuned_pipeline, best_params = self.tune_hyperparameters(
                    X_train, y_train,
                    name,
                    configs[name]['param_grid'],
                    result['pipeline'],
                    scoring='f1'
                )
                
                # Evaluate tuned model on validation
                y_pred = tuned_pipeline.predict(X_val)
                tuned_metrics = self.evaluate_model(y_val, y_pred, average='binary')
                
                # Update results
                results[name]['pipeline'] = tuned_pipeline
                results[name]['val_metrics'] = tuned_metrics
                results[name]['best_params'] = best_params
                
                print(f"  Tuned Validation F1: {tuned_metrics['f1']:.4f}")
        
        # Select best model
        best_name, best_result = self.select_best_model(results, metric='f1')
        
        # Final evaluation on test set
        test_metrics = self.final_evaluation(
            best_result['pipeline'],
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            task_type='binary'
        )
        
        return {
            'task': 'high_mar_prediction',
            'all_models': results,
            'best_model': best_name,
            'best_params': best_result['best_params'],
            'val_metrics': best_result['val_metrics'],
            'test_metrics': test_metrics,
            'splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
    
    def task2_species_classification(
        self,
        feature_cols: List[str],
        min_samples: int = 10,
        include_tuning: bool = True,
        tune_top_n: int = 3
    ) -> Dict:
        """
        Task 2: Multiclass classification to predict bacterial species.
        
        Args:
            feature_cols: List of feature column names (AMR patterns)
            min_samples: Minimum samples per species (group rare species as 'Other')
            include_tuning: Whether to perform hyperparameter tuning
            tune_top_n: Number of top models to tune
            
        Returns:
            Dict: Complete results including all models and final test metrics
        """
        print("="*80)
        print("TASK 2: SPECIES CLASSIFICATION FROM RESISTANCE PROFILE")
        print("="*80)
        
        # Get target
        if 'bacterial_species' not in self.df.columns:
            raise ValueError("bacterial_species column not found")
        
        y = self.df['bacterial_species'].copy()
        X = self.df[feature_cols]
        
        # Group rare species
        species_counts = y.value_counts()
        rare_species = species_counts[species_counts < min_samples].index
        
        if len(rare_species) > 0:
            print(f"\nGrouping {len(rare_species)} rare species as 'Other':")
            for species in rare_species:
                print(f"  - {species}: {species_counts[species]} samples")
            y = y.replace(rare_species, 'Other')
        
        print(f"\nTarget distribution:")
        print(y.value_counts())
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.stratified_split(X, y)
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Train and evaluate all models
        results = self.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='multiclass'
        )
        
        # Hyperparameter tuning for top models
        if include_tuning:
            sorted_models = sorted(
                results.items(),
                key=lambda x: x[1]['val_metrics']['f1'],
                reverse=True
            )
            
            configs = self.get_algorithm_configs()
            
            for name, result in sorted_models[:tune_top_n]:
                tuned_pipeline, best_params = self.tune_hyperparameters(
                    X_train, y_train,
                    name,
                    configs[name]['param_grid'],
                    result['pipeline'],
                    scoring='f1_macro'
                )
                
                # Evaluate tuned model
                y_pred = tuned_pipeline.predict(X_val)
                tuned_metrics = self.evaluate_model(y_val, y_pred, average='macro')
                
                # Update results
                results[name]['pipeline'] = tuned_pipeline
                results[name]['val_metrics'] = tuned_metrics
                results[name]['best_params'] = best_params
                
                print(f"  Tuned Validation F1: {tuned_metrics['f1']:.4f}")
        
        # Select best model
        best_name, best_result = self.select_best_model(results, metric='f1')
        
        # Final evaluation
        test_metrics = self.final_evaluation(
            best_result['pipeline'],
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            task_type='multiclass'
        )
        
        # Add weighted F1 for multiclass
        y_pred_test = best_result['pipeline'].predict(pd.concat([X_train, X_val]))
        y_pred_test = best_result['pipeline'].predict(X_test)
        test_metrics['f1_weighted'] = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        
        return {
            'task': 'species_classification',
            'all_models': results,
            'best_model': best_name,
            'best_params': best_result['best_params'],
            'val_metrics': best_result['val_metrics'],
            'test_metrics': test_metrics,
            'splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
    
    def task3_region_source_classification(
        self,
        feature_cols: List[str],
        target_col: str = 'administrative_region',
        min_samples: int = 10,
        include_tuning: bool = True,
        tune_top_n: int = 3
    ) -> Dict:
        """
        Task 3: Multiclass classification to predict region or source.
        
        Args:
            feature_cols: List of feature column names (AMR + species)
            target_col: Target column ('administrative_region' or 'sample_source')
            min_samples: Minimum samples per class (group rare as 'Other')
            include_tuning: Whether to perform hyperparameter tuning
            tune_top_n: Number of top models to tune
            
        Returns:
            Dict: Complete results including all models and final test metrics
        """
        print("="*80)
        print(f"TASK 3: {target_col.upper().replace('_', ' ')} CLASSIFICATION")
        print("="*80)
        
        # Get target
        if target_col not in self.df.columns:
            raise ValueError(f"{target_col} column not found")
        
        y = self.df[target_col].copy()
        X = self.df[feature_cols]
        
        # Group rare classes
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < min_samples].index
        
        if len(rare_classes) > 0:
            print(f"\nGrouping {len(rare_classes)} rare classes as 'Other':")
            for cls in rare_classes:
                print(f"  - {cls}: {class_counts[cls]} samples")
            y = y.replace(rare_classes, 'Other')
        
        print(f"\nTarget distribution:")
        print(y.value_counts())
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.stratified_split(X, y)
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Train and evaluate all models
        results = self.train_and_evaluate_all_models(
            X_train, X_val, y_train, y_val,
            task_type='multiclass'
        )
        
        # Hyperparameter tuning for top models
        if include_tuning:
            sorted_models = sorted(
                results.items(),
                key=lambda x: x[1]['val_metrics']['f1'],
                reverse=True
            )
            
            configs = self.get_algorithm_configs()
            
            for name, result in sorted_models[:tune_top_n]:
                tuned_pipeline, best_params = self.tune_hyperparameters(
                    X_train, y_train,
                    name,
                    configs[name]['param_grid'],
                    result['pipeline'],
                    scoring='f1_macro'
                )
                
                # Evaluate tuned model
                y_pred = tuned_pipeline.predict(X_val)
                tuned_metrics = self.evaluate_model(y_val, y_pred, average='macro')
                
                # Update results
                results[name]['pipeline'] = tuned_pipeline
                results[name]['val_metrics'] = tuned_metrics
                results[name]['best_params'] = best_params
                
                print(f"  Tuned Validation F1: {tuned_metrics['f1']:.4f}")
        
        # Select best model
        best_name, best_result = self.select_best_model(results, metric='f1')
        
        # Final evaluation
        test_metrics = self.final_evaluation(
            best_result['pipeline'],
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            task_type='multiclass'
        )
        
        # Add weighted F1
        y_pred_test = best_result['pipeline'].predict(X_test)
        test_metrics['f1_weighted'] = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        
        return {
            'task': f'{target_col}_classification',
            'all_models': results,
            'best_model': best_name,
            'best_params': best_result['best_params'],
            'val_metrics': best_result['val_metrics'],
            'test_metrics': test_metrics,
            'splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }


def quick_supervised_analysis(
    df: pd.DataFrame,
    task: str = 'all',
    feature_cols: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """
    Quick function to run supervised analysis tasks.
    
    Args:
        df: Prepared dataframe
        task: 'high_mar', 'species', 'region', 'source', or 'all'
        feature_cols: Feature columns (auto-detected if None)
        **kwargs: Additional arguments passed to task methods
        
    Returns:
        Dict: Results from requested task(s)
    """
    analyzer = SupervisedAMRAnalysis(df)
    results = {}
    
    # Auto-detect features if not provided
    if feature_cols is None:
        binary_cols = [col for col in df.columns if '_binary' in col]
        species_cols = [col for col in df.columns if 'bacterial_species_' in col]
        feature_cols = binary_cols + species_cols
    
    if task == 'high_mar' or task == 'all':
        # Use only AMR features for high MAR prediction
        amr_features = [col for col in feature_cols if '_binary' in col]
        results['high_mar'] = analyzer.task1_high_mar_prediction(amr_features, **kwargs)
    
    if task == 'species' or task == 'all':
        # Use only AMR features for species prediction
        amr_features = [col for col in feature_cols if '_binary' in col]
        results['species'] = analyzer.task2_species_classification(amr_features, **kwargs)
    
    if task == 'region' or task == 'all':
        # Use AMR + species features for region prediction
        results['region'] = analyzer.task3_region_source_classification(
            feature_cols,
            target_col='administrative_region',
            **kwargs
        )
    
    if task == 'source' or task == 'all':
        # Use AMR + species features for source prediction
        results['source'] = analyzer.task3_region_source_classification(
            feature_cols,
            target_col='sample_source',
            **kwargs
        )
    
    return results
