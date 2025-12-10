#!/bin/bash
# Start Streamlit App for AMR Model Deployment
# Usage: ./start_streamlit.sh [port]

PORT=${1:-8501}

echo "=================================="
echo "Starting AMR Streamlit App"
echo "=================================="
echo "Port: $PORT"
echo ""
echo "The app will open in your browser at:"
echo "  http://localhost:$PORT"
echo ""
echo "Press CTRL+C to stop the app"
echo "=================================="
echo ""

streamlit run app.py --server.port $PORT --server.address 0.0.0.0
