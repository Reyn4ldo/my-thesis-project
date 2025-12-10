#!/usr/bin/env python
"""
Test script for FastAPI and Streamlit integration

This script tests:
1. FastAPI server functionality
2. All API endpoints
3. Streamlit app imports
"""

import sys
import time
import subprocess
import requests
import json
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_info(msg):
    print(f"{YELLOW}ℹ {msg}{RESET}")

def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*60)
    print("Testing Python Imports")
    print("="*60)
    
    modules = [
        'fastapi',
        'uvicorn',
        'streamlit',
        'plotly',
        'pydantic',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print_success(f"Import {module}")
        except ImportError as e:
            print_error(f"Import {module}: {e}")
            return False
    
    return True

def test_api_module():
    """Test that API module loads correctly"""
    print("\n" + "="*60)
    print("Testing API Module")
    print("="*60)
    
    try:
        from api import app
        print_success("API module loaded successfully")
        return True
    except Exception as e:
        print_error(f"API module failed to load: {e}")
        return False

def test_streamlit_module():
    """Test that Streamlit app module loads correctly"""
    print("\n" + "="*60)
    print("Testing Streamlit Module")
    print("="*60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('app', 'app.py')
        app_module = importlib.util.module_from_spec(spec)
        print_success("Streamlit app module validated")
        return True
    except Exception as e:
        print_error(f"Streamlit app module failed: {e}")
        return False

def test_api_endpoints():
    """Test FastAPI endpoints"""
    print("\n" + "="*60)
    print("Testing FastAPI Endpoints")
    print("="*60)
    
    # Start server
    print_info("Starting FastAPI server...")
    server_process = subprocess.Popen(
        ['uvicorn', 'api:app', '--host', '0.0.0.0', '--port', '8000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    
    BASE_URL = "http://localhost:8000"
    
    try:
        # Test 1: Root endpoint
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print_success("GET / (root)")
        else:
            print_error(f"GET / failed: {response.status_code}")
        
        # Test 2: Health check
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print_success("GET /health")
        else:
            print_error(f"GET /health failed: {response.status_code}")
        
        # Test 3: List models
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print_success(f"GET /models (found {data['count']} models)")
        else:
            print_error(f"GET /models failed: {response.status_code}")
        
        # Test 4: Model info (only if model exists)
        if Path("high_MAR_model.pkl").exists():
            response = requests.post(
                f"{BASE_URL}/models/info",
                json={"model_path": "high_MAR_model.pkl"}
            )
            if response.status_code == 200:
                print_success("POST /models/info")
            else:
                print_error(f"POST /models/info failed: {response.status_code}")
        else:
            print_info("Skipping model-specific tests (no model found)")
        
        result = True
        
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to API server")
        result = False
    except Exception as e:
        print_error(f"API test failed: {e}")
        result = False
    finally:
        # Stop server
        print_info("Stopping FastAPI server...")
        server_process.terminate()
        server_process.wait(timeout=5)
    
    return result

def test_files():
    """Test that all required files exist"""
    print("\n" + "="*60)
    print("Testing File Structure")
    print("="*60)
    
    required_files = [
        'api.py',
        'app.py',
        'requirements.txt',
        'start_api.sh',
        'start_streamlit.sh',
        'WEB_DEPLOYMENT_GUIDE.md',
        'README.md',
        'DEPLOYMENT_GUIDE.md',
        'model_deployment.py'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print_success(f"File exists: {file}")
        else:
            print_error(f"File missing: {file}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FastAPI & Streamlit Integration Test Suite")
    print("="*60)
    
    results = {
        "File Structure": test_files(),
        "Python Imports": test_imports(),
        "API Module": test_api_module(),
        "Streamlit Module": test_streamlit_module(),
        "API Endpoints": test_api_endpoints()
    }
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print_success("All tests passed!")
        print("="*60)
        print("\nYou can now start the applications:")
        print("  FastAPI:   ./start_api.sh")
        print("  Streamlit: ./start_streamlit.sh")
        return 0
    else:
        print_error("Some tests failed!")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
