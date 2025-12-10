# Quick Start Guide - Web Deployment

This guide helps you get started with the FastAPI and Streamlit web deployment features in under 5 minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **Trained AMR model** (`.pkl` file) - If you don't have one, see "Training a Model" below
3. **Dependencies installed** - Run: `pip install -r requirements.txt`

## Option 1: Streamlit Web UI (Easiest)

For interactive, user-friendly web interface:

```bash
# Start the Streamlit app
./start_streamlit.sh
# Or: streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**What you can do:**
- View model information and performance metrics
- Make single predictions with an interactive form
- Upload CSV files for batch predictions
- Visualize results with charts
- Download prediction results

## Option 2: FastAPI REST API (For Programmatic Access)

For REST API integration:

```bash
# Start the FastAPI server
./start_api.sh
# Or: uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**Access interactive documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Example API call:**
```bash
# Check health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Make a prediction (Python)
import requests
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'features': {...},  # Your feature dictionary
        'model_path': 'high_MAR_model.pkl'
    }
)
print(response.json())
```

## Training a Model (If Needed)

If you don't have a trained model yet:

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

# Train and save model
analyzer = SupervisedAMRAnalysis(df)
results = analyzer.task1_high_mar_prediction(
    feature_cols=feature_cols,
    threshold=0.3,
    include_tuning=False,  # Set True for better performance (slower)
    save_model_path='high_MAR_model.pkl'
)
```

This creates:
- `high_MAR_model.pkl` - The trained model
- `high_MAR_model_metadata.json` - Model documentation

## Testing Your Setup

Run the test suite to verify everything works:

```bash
python test_web_integration.py
```

This will test:
- Python imports
- API module loading
- Streamlit app loading
- All API endpoints
- File structure

## Common Use Cases

### 1. Clinical Decision Support
Use Streamlit for real-time patient assessments:
1. Open Streamlit app
2. Enter patient's antibiotic resistance profile
3. Get instant prediction with confidence scores

### 2. Batch Lab Processing
Use FastAPI for automated processing:
```python
import requests

with open('lab_results.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/batch',
        files={'file': f},
        data={'model_path': 'high_MAR_model.pkl'}
    )
    
with open('predictions.csv', 'wb') as f:
    f.write(response.content)
```

### 3. Integration with EHR Systems
Use FastAPI endpoints in your application:
```python
def check_patient_resistance(patient_data):
    response = requests.post(
        'http://localhost:8000/predict',
        json={
            'features': patient_data,
            'model_path': 'high_MAR_model.pkl'
        }
    )
    return response.json()
```

## Next Steps

For detailed documentation:
- **WEB_DEPLOYMENT_GUIDE.md** - Comprehensive guide with examples
- **README.md** - Complete project documentation
- **DEPLOYMENT_GUIDE.md** - General deployment scenarios

## Troubleshooting

**Port already in use?**
```bash
# Use a different port
./start_api.sh 8080
./start_streamlit.sh 8502
```

**Module not found errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**No models found?**
- Ensure `.pkl` files are in the current directory
- Check that metadata `.json` files exist alongside models

**API connection refused?**
- Verify the server is running
- Check the correct port is being used
- Ensure no firewall blocking

## Support

Need help? Check:
1. WEB_DEPLOYMENT_GUIDE.md for detailed instructions
2. API documentation at http://localhost:8000/docs
3. Test suite output: `python test_web_integration.py`

---

**Built with:**
- FastAPI (REST API framework)
- Streamlit (Web UI framework)
- Plotly (Visualizations)
- Pydantic (Data validation)

**Project:** AMR Analysis Thesis - Antimicrobial Resistance Prediction
