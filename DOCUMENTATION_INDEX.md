# Documentation Index

This file helps you quickly find the right documentation for your needs.

## 🚀 Getting Started

- **[QUICKSTART_WEB.md](QUICKSTART_WEB.md)** - Start here! 5-minute setup guide for FastAPI and Streamlit
- **[README.md](README.md)** - Complete project overview and pipeline documentation
- **[Installation](#installation)** - See README.md section for dependency installation

## 🌐 Web Deployment

- **[WEB_DEPLOYMENT_GUIDE.md](WEB_DEPLOYMENT_GUIDE.md)** - Comprehensive guide for FastAPI and Streamlit
  - FastAPI REST API documentation
  - Streamlit web UI guide
  - Usage examples
  - Troubleshooting
  
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - General model deployment scenarios
  - Command-line deployment
  - FastAPI REST API scenario
  - Streamlit web application scenario
  - Surveillance dashboard examples

## 📚 Core Documentation

- **[README.md](README.md)** - Main documentation covering:
  - Phase 1: Data Preparation
  - Phase 2: Unsupervised Learning
  - Phase 3: Supervised Learning
  - Phase 4: Model Deployment
  - Quick Start examples

- **[THESIS_REPORT.md](THESIS_REPORT.md)** - Academic thesis report
  - Research methodology
  - Results and analysis
  - Performance metrics

## 🔧 Technical Reference

### API Documentation (When Running)
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Scripts
- **[start_api.sh](start_api.sh)** - Start FastAPI server
- **[start_streamlit.sh](start_streamlit.sh)** - Start Streamlit app
- **[test_web_integration.py](test_web_integration.py)** - Integration test suite

### Core Modules
- **[api.py](api.py)** - FastAPI REST API implementation
- **[app.py](app.py)** - Streamlit web application
- **[model_deployment.py](model_deployment.py)** - Model deployment utilities
- **[data_preparation.py](data_preparation.py)** - Data preprocessing
- **[supervised_analysis.py](supervised_analysis.py)** - Supervised ML
- **[unsupervised_analysis.py](unsupervised_analysis.py)** - Unsupervised ML

## 📖 By Use Case

### I want to make predictions through a web interface
→ [QUICKSTART_WEB.md](QUICKSTART_WEB.md) → Streamlit section

### I want to integrate predictions into my application
→ [WEB_DEPLOYMENT_GUIDE.md](WEB_DEPLOYMENT_GUIDE.md) → FastAPI section

### I want to understand the complete pipeline
→ [README.md](README.md)

### I want to train a new model
→ [README.md](README.md) → Phase 3: Supervised Pattern Recognition

### I want to deploy a model in production
→ [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### I want to understand the research
→ [THESIS_REPORT.md](THESIS_REPORT.md)

## 🎯 Quick Actions

| I want to... | Command | Documentation |
|--------------|---------|---------------|
| Start web UI | `./start_streamlit.sh` | [QUICKSTART_WEB.md](QUICKSTART_WEB.md) |
| Start API | `./start_api.sh` | [QUICKSTART_WEB.md](QUICKSTART_WEB.md) |
| Run tests | `python test_web_integration.py` | [test_web_integration.py](test_web_integration.py) |
| Train model | See code in README | [README.md](README.md) |
| Make predictions | Use Streamlit or API | [WEB_DEPLOYMENT_GUIDE.md](WEB_DEPLOYMENT_GUIDE.md) |

## 🆘 Troubleshooting

1. **Installation issues** → [README.md](README.md) Installation section
2. **Web deployment issues** → [WEB_DEPLOYMENT_GUIDE.md](WEB_DEPLOYMENT_GUIDE.md) Troubleshooting section
3. **API errors** → Check http://localhost:8000/docs for API status
4. **Model not found** → Ensure `.pkl` files are in current directory

## 📦 File Overview

```
my-thesis-project/
├── 📘 README.md                     # Main documentation
├── 🚀 QUICKSTART_WEB.md             # Quick start guide
├── 🌐 WEB_DEPLOYMENT_GUIDE.md       # Web deployment guide
├── 📋 DEPLOYMENT_GUIDE.md           # General deployment
├── 📊 THESIS_REPORT.md              # Research report
├── 📑 DOCUMENTATION_INDEX.md        # This file
│
├── 🔌 api.py                        # FastAPI REST API
├── 🎨 app.py                        # Streamlit web UI
├── 📦 model_deployment.py           # Deployment utilities
├── 🧬 data_preparation.py           # Data preprocessing
├── 🤖 supervised_analysis.py        # Supervised ML
├── 🔍 unsupervised_analysis.py      # Unsupervised ML
│
├── 🚀 start_api.sh                  # Start FastAPI
├── 🚀 start_streamlit.sh            # Start Streamlit
├── 🧪 test_web_integration.py       # Integration tests
├── 📝 requirements.txt              # Dependencies
└── 📊 rawdata.csv                   # Sample data
```

## 💡 Tips

- Start with [QUICKSTART_WEB.md](QUICKSTART_WEB.md) for the fastest path to deployment
- Use [WEB_DEPLOYMENT_GUIDE.md](WEB_DEPLOYMENT_GUIDE.md) as your primary reference
- Keep [README.md](README.md) open for complete pipeline understanding
- Check API documentation at `/docs` when the server is running

## 🔗 External Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://docs.streamlit.io/
- **Scikit-learn**: https://scikit-learn.org/

---

**Need Help?** Start with the documentation matching your goal from the "By Use Case" section above.
