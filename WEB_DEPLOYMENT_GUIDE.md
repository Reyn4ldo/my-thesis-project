# Web Deployment Guide - FastAPI & Streamlit

This guide explains how to use the FastAPI REST API and Streamlit web interface for AMR model deployment.

## Table of Contents

1. [Installation](#installation)
2. [FastAPI REST API](#fastapi-rest-api)
3. [Streamlit Web Application](#streamlit-web-application)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)

---

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Or install web frameworks separately:

```bash
pip install fastapi uvicorn streamlit plotly python-multipart
```

---

## FastAPI REST API

### Starting the API Server

**Option 1: Using the startup script**
```bash
./start_api.sh
# Or specify a custom port
./start_api.sh 8080
```

**Option 2: Using uvicorn directly**
```bash
uvicorn api:app --reload --port 8000
```

**Option 3: Using Python**
```bash
python api.py
```

The API server will start at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
  - Interactive API documentation
  - Try out endpoints directly in the browser
  - View request/response schemas

- **ReDoc**: http://localhost:8000/redoc
  - Alternative documentation format
  - Clean, readable interface

- **OpenAPI JSON**: http://localhost:8000/openapi.json
  - Raw OpenAPI specification

### Available Endpoints

#### 1. API Root
```bash
GET http://localhost:8000/
```
Returns API information and available endpoints.

#### 2. Health Check
```bash
GET http://localhost:8000/health
```
Check if the API is running and how many models are loaded in cache.

#### 3. List Available Models
```bash
GET http://localhost:8000/models
```
Returns a list of all `.pkl` model files in the current directory.

#### 4. Get Model Information
```bash
POST http://localhost:8000/models/info
Content-Type: application/json

{
  "model_path": "high_MAR_model.pkl"
}
```
Returns detailed information about a specific model including metrics and features.

#### 5. Single Isolate Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "features": {
    "ampicillin_binary": 1.0,
    "gentamicin_binary": 0.0,
    "ciprofloxacin_binary": 0.0,
    ...
  },
  "model_path": "high_MAR_model.pkl",
  "return_proba": true
}
```
Returns prediction and probabilities for a single isolate.

#### 6. Batch Prediction from CSV
```bash
POST http://localhost:8000/predict/batch
Content-Type: multipart/form-data

file: [CSV file]
model_path: high_MAR_model.pkl
include_proba: true
include_original: true
```
Upload a CSV file and get predictions for all samples.

### Using the API with Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. List models
response = requests.get(f"{BASE_URL}/models")
print(response.json())

# 3. Get model info
response = requests.post(
    f"{BASE_URL}/models/info",
    json={"model_path": "high_MAR_model.pkl"}
)
info = response.json()
print(f"Model: {info['model_info']['model_type']}")
print(f"Test F1: {info['metrics']['test']['f1']:.4f}")

# 4. Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "features": {
            "ampicillin_binary": 1.0,
            "gentamicin_binary": 0.0,
            # ... all required features
        },
        "model_path": "high_MAR_model.pkl",
        "return_proba": True
    }
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability_class_1']:.2%}")

# 5. Batch prediction
with open('new_isolates.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'model_path': 'high_MAR_model.pkl',
        'include_proba': 'true',
        'include_original': 'true'
    }
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        files=files,
        data=data
    )

# Save predictions
with open('predictions.csv', 'wb') as f:
    f.write(response.content)
```

### Using the API with curl

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Model info
curl -X POST http://localhost:8000/models/info \
  -H "Content-Type: application/json" \
  -d '{"model_path": "high_MAR_model.pkl"}'

# Single prediction (see API docs for full feature list)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {...},
    "model_path": "high_MAR_model.pkl",
    "return_proba": true
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@new_isolates.csv" \
  -F "model_path=high_MAR_model.pkl" \
  -F "include_proba=true" \
  -o predictions.csv
```

---

## Streamlit Web Application

### Starting the Streamlit App

**Option 1: Using the startup script**
```bash
./start_streamlit.sh
# Or specify a custom port
./start_streamlit.sh 8502
```

**Option 2: Using streamlit directly**
```bash
streamlit run app.py
```

**Option 3: With custom configuration**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The app will automatically open in your browser at `http://localhost:8501`

### Application Features

The Streamlit app provides three main tabs:

#### 1. Model Info Tab
- View model details (task name, model type, creation date)
- Performance metrics (accuracy, precision, recall, F1)
- Model hyperparameters
- List of all required features

#### 2. Single Prediction Tab
- Interactive form with dropdowns for each antibiotic
- Select Susceptible/Intermediate (0) or Resistant (1) for each antibiotic
- Click "Predict" to get:
  - Prediction result (Low MAR or High MAR)
  - Confidence scores for both classes
  - Visual probability chart

#### 3. Batch Prediction Tab
- Upload CSV file with multiple isolates
- Preview data before prediction
- Options to include/exclude probabilities and original columns
- Results summary with statistics
- Interactive pie chart of prediction distribution
- View all predictions in a table
- Download predictions as CSV

### Using the Streamlit App

1. **Select a Model**
   - Use the sidebar dropdown to select a model file
   - The app will load the model and display its information

2. **Make Single Predictions**
   - Go to "Single Prediction" tab
   - Fill out the form with resistance values for each antibiotic
   - Click "Predict" to see results
   - View confidence scores and probability chart

3. **Batch Predictions**
   - Go to "Batch Prediction" tab
   - Upload a CSV file with your data
   - Preview the data to ensure it's correct
   - Click "Make Predictions"
   - View summary statistics and visualizations
   - Download results as CSV

### CSV File Format

For batch predictions, your CSV file must include all required feature columns:

```csv
ampicillin_binary,gentamicin_binary,ciprofloxacin_binary,...
1,0,0,...
0,1,0,...
0,0,1,...
```

- Column names must match the model's required features exactly
- Values should be 0 (Susceptible/Intermediate) or 1 (Resistant)
- Missing values will be handled by the model's preprocessing pipeline

---

## Usage Examples

### Example 1: Clinical Decision Support System

Use the Streamlit app for real-time clinical decisions:

1. Start Streamlit: `./start_streamlit.sh`
2. Load your trained model
3. Enter patient's antibiotic resistance profile
4. Get instant prediction with confidence scores
5. Use results to guide treatment decisions

### Example 2: Batch Processing Lab Results

Use FastAPI for automated batch processing:

```python
import requests
import glob

API_URL = "http://localhost:8000"

# Process all new lab result files
for csv_file in glob.glob("lab_results_*.csv"):
    with open(csv_file, 'rb') as f:
        response = requests.post(
            f"{API_URL}/predict/batch",
            files={'file': f},
            data={'model_path': 'high_MAR_model.pkl'}
        )
    
    # Save predictions
    output_file = csv_file.replace('lab_results', 'predictions')
    with open(output_file, 'wb') as f:
        f.write(response.content)
    
    print(f"Processed {csv_file} -> {output_file}")
```

### Example 3: Integration with Electronic Health Records (EHR)

```python
import requests

def check_patient_amr(patient_id, resistance_profile):
    """Check AMR risk for a patient"""
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "features": resistance_profile,
            "model_path": "high_MAR_model.pkl",
            "return_proba": True
        }
    )
    
    result = response.json()
    
    # Log to EHR
    log_amr_assessment(
        patient_id=patient_id,
        prediction=result['prediction'],
        confidence=result['probability_class_1'],
        timestamp=datetime.now()
    )
    
    # Alert if high risk
    if result['prediction'] == 1 and result['probability_class_1'] > 0.7:
        send_alert_to_provider(patient_id, result)
    
    return result
```

### Example 4: Surveillance Dashboard

Use FastAPI to power a monitoring dashboard:

```python
import requests
import pandas as pd
import plotly.express as px

# Get predictions for current month
response = requests.post(
    "http://localhost:8000/predict/batch",
    files={'file': open('monthly_isolates.csv', 'rb')},
    data={'model_path': 'high_MAR_model.pkl'}
)

# Analyze results
df = pd.read_csv(io.StringIO(response.text))
high_mar_rate = (df['high_mar_prediction_prediction'] == 1).mean()

print(f"High MAR Rate: {high_mar_rate:.1%}")

# Visualize trends
fig = px.line(df.groupby('date')['high_mar_prediction_prediction'].mean())
fig.show()
```

---

## Troubleshooting

### FastAPI Issues

**Port already in use**
```bash
# Use a different port
uvicorn api:app --port 8001
```

**Model not found**
- Ensure the .pkl file exists in the current directory
- Check the model path in your request
- Use `GET /models` to see available models

**Feature validation error**
- Ensure your input includes all required features
- Use `POST /models/info` to see required features
- Check that feature names match exactly (case-sensitive)

### Streamlit Issues

**Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Model not loading**
- Check that .pkl files exist in the current directory
- Ensure metadata JSON files are present
- Clear Streamlit cache: `streamlit cache clear`

**CSV upload fails**
- Verify CSV has all required feature columns
- Check for correct column names (case-sensitive)
- Ensure values are numeric (0 or 1)

### General Issues

**Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Performance issues**
- Model is cached after first load
- Consider reducing batch sizes for large CSV files
- Use FastAPI for production deployments (better performance)

**Connection refused**
- Ensure the server is running
- Check the correct port is being used
- Verify firewall settings

---

## Best Practices

### Security

1. **Authentication**: Add authentication for production deployments
2. **HTTPS**: Use HTTPS in production
3. **Input Validation**: Always validate input data
4. **Rate Limiting**: Implement rate limiting for API endpoints

### Performance

1. **Model Caching**: Models are cached after first load
2. **Batch Processing**: Use batch endpoints for multiple predictions
3. **Async Processing**: Consider async processing for large files
4. **Monitoring**: Monitor API response times and errors

### Deployment

1. **Environment Variables**: Use environment variables for configuration
2. **Docker**: Consider containerizing the applications
3. **Load Balancing**: Use load balancer for high-traffic scenarios
4. **Logging**: Implement comprehensive logging

---

## Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Project README**: See README.md for complete pipeline documentation
- **Deployment Guide**: See DEPLOYMENT_GUIDE.md for additional deployment scenarios

---

## Support

For issues or questions:
1. Check this guide first
2. Review the main README.md
3. Check API documentation at /docs
4. Review example scripts in the repository
