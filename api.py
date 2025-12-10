"""
FastAPI Backend for AMR Model Deployment

This API provides RESTful endpoints for deploying trained AMR models.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /               - API information
    GET  /health         - Health check
    GET  /models         - List available models
    POST /models/info    - Get model information
    POST /predict        - Single isolate prediction
    POST /predict/batch  - Batch prediction from CSV

Author: Thesis Project
Date: December 2024
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import io
import json
from pathlib import Path
from model_deployment import ModelDeployment
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="AMR Model Deployment API",
    description="REST API for Antimicrobial Resistance model predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model_cache: Dict[str, ModelDeployment] = {}


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single isolate prediction"""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and values",
        example={
            "ampicillin_binary": 1.0,
            "gentamicin_binary": 0.0,
            "ciprofloxacin_binary": 0.0
        }
    )
    model_path: str = Field(
        default="high_MAR_model.pkl",
        description="Path to the model file"
    )
    return_proba: bool = Field(
        default=True,
        description="Whether to return prediction probabilities"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Predicted class")
    probability_class_0: Optional[float] = Field(None, description="Probability of class 0")
    probability_class_1: Optional[float] = Field(None, description="Probability of class 1")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")


class ModelInfoRequest(BaseModel):
    """Request model for getting model information"""
    model_path: str = Field(
        default="high_MAR_model.pkl",
        description="Path to the model file"
    )


def load_model(model_path: str) -> ModelDeployment:
    """
    Load a model from cache or disk.
    
    Args:
        model_path: Path to model file
        
    Returns:
        ModelDeployment instance
    """
    if model_path not in _model_cache:
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}"
            )
        try:
            _model_cache[model_path] = ModelDeployment(model_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    return _model_cache[model_path]


@app.get("/")
async def root():
    """API root - returns basic information"""
    return {
        "name": "AMR Model Deployment API",
        "version": "1.0.0",
        "description": "REST API for Antimicrobial Resistance predictions",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "POST /models/info": "Get model information",
            "POST /predict": "Single isolate prediction",
            "POST /predict/batch": "Batch prediction from CSV"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(_model_cache)
    }


@app.get("/models")
async def list_models():
    """List available model files in the current directory"""
    model_files = list(Path(".").glob("*.pkl"))
    models = []
    
    for model_file in model_files:
        model_info = {
            "name": model_file.name,
            "path": str(model_file),
            "size_bytes": model_file.stat().st_size
        }
        
        # Check if metadata exists
        metadata_path = model_file.with_suffix('.pkl').name.replace('.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            model_info["has_metadata"] = True
        else:
            model_info["has_metadata"] = False
            
        models.append(model_info)
    
    return {
        "models": models,
        "count": len(models)
    }


@app.post("/models/info")
async def get_model_info(request: ModelInfoRequest):
    """Get detailed information about a specific model"""
    try:
        deployment = load_model(request.model_path)
        
        info = deployment.get_model_info()
        metrics = deployment.get_performance_metrics()
        features = deployment.get_required_features()
        
        return {
            "model_info": info,
            "metrics": metrics,
            "features": {
                "count": len(features),
                "names": features[:10] if len(features) > 10 else features,
                "total": len(features)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Make a prediction for a single isolate.
    
    Args:
        request: PredictionRequest with features and model_path
        
    Returns:
        PredictionResponse with prediction and probabilities
    """
    try:
        deployment = load_model(request.model_path)
        
        # Make prediction
        result = deployment.predict_single(
            request.features,
            return_proba=request.return_proba
        )
        
        # Add model info
        result['model_info'] = deployment.get_model_info()
        
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        )


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with isolate data"),
    model_path: str = Body(default="high_MAR_model.pkl"),
    include_proba: bool = Body(default=True),
    include_original: bool = Body(default=True)
):
    """
    Make predictions for multiple isolates from a CSV file.
    
    Args:
        file: Uploaded CSV file
        model_path: Path to model file
        include_proba: Include prediction probabilities
        include_original: Include original columns in output
        
    Returns:
        CSV file with predictions
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV file"
            )
        
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Load model
        deployment = load_model(model_path)
        
        # Make predictions
        if include_proba:
            predictions, probabilities = deployment.predict(df, include_proba=True)
        else:
            predictions = deployment.predict(df, include_proba=False)
            probabilities = None
        
        # Create results dataframe
        if include_original:
            results = df.copy()
        else:
            results = pd.DataFrame()
        
        # Add predictions
        task_name = deployment.metadata['model_info']['task_name'] if deployment.metadata else 'prediction'
        results[f'{task_name}_prediction'] = predictions
        
        # Add probabilities if available
        if probabilities is not None:
            if probabilities.shape[1] == 2:  # Binary classification
                results[f'{task_name}_probability_class_0'] = probabilities[:, 0]
                results[f'{task_name}_probability_class_1'] = probabilities[:, 1]
            else:  # Multiclass
                for i in range(probabilities.shape[1]):
                    results[f'{task_name}_probability_class_{i}'] = probabilities[:, i]
        
        # Convert to CSV
        output = io.StringIO()
        results.to_csv(output, index=False)
        output.seek(0)
        
        # Return as streaming response
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{file.filename}"
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
