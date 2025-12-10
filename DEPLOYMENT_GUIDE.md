# AMR Model Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying trained AMR (Antimicrobial Resistance) models for high MAR/MDR prediction.

## Quick Start

### 1. Train and Save a Model

```python
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis

# Prepare data
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(include_binary=True, include_ordinal=False, 
                       include_onehot=True, scale=False, drop_original_int=True)

# Get features
groups = prep.get_feature_groups()
feature_cols = groups['binary_resistance']

# Train and save model (auto-save enabled)
analyzer = SupervisedAMRAnalysis(df)
results = analyzer.task1_high_mar_prediction(
    feature_cols=feature_cols,
    threshold=0.3,
    include_tuning=True,
    tune_top_n=3,
    save_model_path='high_MAR_model.pkl'  # Auto-save
)
```

**Output files:**
- `high_MAR_model.pkl` - Trained model pipeline
- `high_MAR_model_metadata.json` - Model documentation

### 2. Deploy the Model

#### Option A: Command-line Deployment

```bash
# View model info
python deploy_model.py --model high_MAR_model.pkl --info

# Make predictions
python deploy_model.py \
    --model high_MAR_model.pkl \
    --input new_isolates.csv \
    --output predictions.csv
```

#### Option B: Python API

```python
from model_deployment import ModelDeployment

# Load model
deployment = ModelDeployment('high_MAR_model.pkl')

# Batch prediction
results = deployment.predict_from_csv(
    input_csv='new_isolates.csv',
    output_csv='predictions.csv',
    include_proba=True
)

# Single prediction
features = {...}  # Antibiotic resistance pattern
result = deployment.predict_single(features, return_proba=True)
```

## Model Files

### Model Pipeline (.pkl)
Contains the complete trained pipeline including:
- Data preprocessing (imputation, scaling)
- Feature transformation
- Trained classifier with optimized hyperparameters

### Model Metadata (.json)
Contains comprehensive documentation:
```json
{
  "model_info": {
    "task_name": "high_mar_prediction",
    "model_type": "LogisticRegression",
    "hyperparameters": {...},
    "created_at": "2025-12-09T..."
  },
  "features": {
    "feature_columns": [...],
    "num_features": 23,
    "feature_format": "Binary resistance encoding (R=1, S/I=0)"
  },
  "metrics": {
    "test": {
      "accuracy": 0.9756,
      "precision": 0.8571,
      "recall": 0.8571,
      "f1": 0.8571
    }
  }
}
```

## Input Data Format

New data must include all required features as binary columns:

```csv
ampicillin_binary,gentamicin_binary,ciprofloxacin_binary,...
1,0,0,...
0,1,0,...
```

**Requirements:**
- All feature columns present (check with `deployment.get_required_features()`)
- Binary values (0 = S/I, 1 = R)
- CSV format with headers
- Missing values will be imputed by the pipeline

## Output Format

Predictions include:

```csv
ampicillin_binary,...,high_mar_prediction_prediction,probability_class_0,probability_class_1
1,...,1,0.12,0.88
0,...,0,0.91,0.09
```

**Columns:**
- Original input features (optional)
- `*_prediction`: Predicted class (0 = Low MAR, 1 = High MAR)
- `probability_class_0`: Probability of Low MAR
- `probability_class_1`: Probability of High MAR

## Deployment Scenarios

### Scenario 1: Batch Processing
Process lab results periodically:

```python
from model_deployment import predict_from_csv
import datetime

# Daily processing
date_str = datetime.date.today().strftime('%Y%m%d')
predict_from_csv(
    model_path='high_MAR_model.pkl',
    input_csv=f'lab_results_{date_str}.csv',
    output_csv=f'predictions_{date_str}.csv'
)
```

### Scenario 2: FastAPI REST API
Real-time decision support with modern REST API:

```bash
# Start FastAPI server
uvicorn api:app --reload --port 8000
```

**Features:**
- RESTful API endpoints
- Automatic OpenAPI documentation
- Request/response validation
- Model caching for performance
- CORS support for frontend integration

**Example Usage:**

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'features': {
            'ampicillin_binary': 1.0,
            'gentamicin_binary': 0.0,
            'ciprofloxacin_binary': 0.0,
            # ... all required features
        },
        'model_path': 'high_MAR_model.pkl',
        'return_proba': True
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability_class_1']:.2%}")

# Batch prediction with CSV upload
with open('new_isolates.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/predict/batch',
        files=files,
        data={'model_path': 'high_MAR_model.pkl'}
    )
    
# Save predictions
with open('predictions.csv', 'wb') as f:
    f.write(response.content)
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Scenario 3: Streamlit Web Application
Interactive web interface for non-technical users:

```bash
# Start Streamlit app
streamlit run app.py
```

**Features:**
- 📊 Model information dashboard
- 🔬 Single isolate prediction form
- 📁 CSV upload for batch predictions
- 📈 Interactive visualizations
- ⬇️ Download prediction results

**Use Cases:**
- Clinical decision support
- Laboratory information systems
- Epidemiological surveillance
- Educational demonstrations

### Scenario 4: Surveillance Dashboard
Monitor trends over time:

```python
import pandas as pd
from model_deployment import ModelDeployment

model = ModelDeployment('high_MAR_model.pkl')

# Process monthly data
for month_file in monthly_files:
    df = pd.read_csv(month_file)
    predictions = model.predict(df, include_proba=True)
    
    # Aggregate by region
    df['prediction'] = predictions
    summary = df.groupby('region')['prediction'].agg(['sum', 'count', 'mean'])
    
    # Alert on high MDR rates
    high_mdr_regions = summary[summary['mean'] > 0.15]
    if not high_mdr_regions.empty:
        send_alert(high_mdr_regions)
```

## Best Practices

### 1. Model Versioning
Use descriptive filenames with version/date:
```
high_MAR_model_v1.0_20250609.pkl
high_MAR_model_production_v2.1.pkl
```

### 2. Validation Before Deployment
Always review metadata before deploying:
```python
deployment = ModelDeployment('model.pkl')
info = deployment.get_model_info()
metrics = deployment.get_performance_metrics()
features = deployment.get_required_features()
```

### 3. Error Handling
Wrap predictions in try-except:
```python
try:
    results = deployment.predict_from_csv(input_csv, output_csv)
except ValueError as e:
    print(f"Feature validation error: {e}")
except Exception as e:
    print(f"Prediction error: {e}")
```

### 4. Performance Monitoring
Track prediction distribution over time:
```python
predictions = df['prediction'].value_counts(normalize=True)
if predictions.get(1, 0) > 0.20:  # > 20% high MAR
    alert("Unusually high MDR rate detected")
```

### 5. Regular Retraining
- Monitor for concept drift
- Retrain quarterly with new data
- Compare new vs. old model performance
- Update deployment when improvement is significant

## Testing

Test deployment before production:

```bash
# Run unit tests
python -m unittest test_model_deployment -v

# Test on sample data
python deploy_model.py --model high_MAR_model.pkl --input test_data.csv --output test_predictions.csv

# Verify outputs
head test_predictions.csv
```

## Troubleshooting

### Missing Features Error
```
ValueError: Missing required features: ['ampicillin_binary', ...]
```
**Solution:** Ensure input data includes all required features from `get_required_features()`

### Low Confidence Predictions
Many predictions with probability ~0.5
**Solution:** Review feature quality, consider retraining with more data

### Authentication Error (Git)
```
fatal: Authentication failed
```
**Solution:** Use `report_progress` tool instead of direct git commands

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review examples in `examples_deployment.py`
3. Run the complete workflow: `python complete_workflow.py`
4. Check test cases in `test_model_deployment.py`

## License

Part of the AMR Analysis Thesis Project - December 2025
