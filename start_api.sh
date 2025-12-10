#!/bin/bash
# Start FastAPI Server for AMR Model Deployment
# Usage: ./start_api.sh [port]

PORT=${1:-8000}

echo "=================================="
echo "Starting AMR API Server"
echo "=================================="
echo "Port: $PORT"
echo ""
echo "API Documentation will be available at:"
echo "  - http://localhost:$PORT/docs (Swagger UI)"
echo "  - http://localhost:$PORT/redoc (ReDoc)"
echo ""
echo "Press CTRL+C to stop the server"
echo "=================================="
echo ""

uvicorn api:app --host 0.0.0.0 --port $PORT --reload
