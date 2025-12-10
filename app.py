"""
Streamlit Web Application for AMR Model Deployment

This application provides a user-friendly interface for:
1. Viewing model information
2. Making single isolate predictions
3. Batch predictions from CSV files
4. Visualizing prediction results

Usage:
    streamlit run app.py

Author: Thesis Project
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from model_deployment import ModelDeployment
import io

# Page configuration
st.set_page_config(
    page_title="AMR Model Deployment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached(model_path):
    """Load and cache the model"""
    return ModelDeployment(model_path)


def find_available_models():
    """Find all available model files"""
    model_files = list(Path(".").glob("*.pkl"))
    return [str(f) for f in model_files]


def display_model_info(deployment):
    """Display model information in a structured way"""
    st.subheader("📊 Model Information")
    
    info = deployment.get_model_info()
    metrics = deployment.get_performance_metrics()
    features = deployment.get_required_features()
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Task", info.get('task_name', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", info.get('model_type', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Number of Features", len(features))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display metrics
    if 'test' in metrics and metrics['test']:
        st.subheader("🎯 Model Performance (Test Set)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['test'].get('accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics['test'].get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics['test'].get('recall', 0):.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics['test'].get('f1', 0):.4f}")
    
    # Display hyperparameters
    with st.expander("🔧 Model Hyperparameters"):
        if 'hyperparameters' in info:
            st.json(info['hyperparameters'])
        else:
            st.info("No hyperparameters available")
    
    # Display features
    with st.expander("📋 Required Features"):
        st.write(f"Total features: {len(features)}")
        st.dataframe(pd.DataFrame({'Feature Name': features}), use_container_width=True)


def single_prediction_tab(deployment):
    """Tab for single isolate prediction"""
    st.subheader("🔬 Single Isolate Prediction")
    
    st.markdown("""
    Enter the antibiotic resistance profile for a single isolate.
    - **0**: Susceptible (S) or Intermediate (I)
    - **1**: Resistant (R)
    """)
    
    features = deployment.get_required_features()
    
    # Create input form
    st.markdown("### Enter Resistance Profile")
    
    # Organize features in columns
    num_cols = 3
    cols = st.columns(num_cols)
    
    feature_values = {}
    for idx, feature in enumerate(features):
        with cols[idx % num_cols]:
            # Create a more user-friendly label
            label = feature.replace('_binary', '').replace('_', ' ').title()
            feature_values[feature] = st.selectbox(
                label,
                options=[0, 1],
                format_func=lambda x: "Susceptible/Intermediate" if x == 0 else "Resistant",
                key=f"feature_{feature}"
            )
    
    # Prediction button
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        try:
            with st.spinner("Making prediction..."):
                result = deployment.predict_single(feature_values, return_proba=True)
            
            # Display results
            st.markdown("---")
            st.subheader("📈 Prediction Results")
            
            prediction = result['prediction']
            
            if prediction == 1:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("### ⚠️ **High MAR/MDR Predicted**")
                st.markdown("This isolate is predicted to have high multidrug resistance.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### ✅ **Low MAR Predicted**")
                st.markdown("This isolate is predicted to have low multidrug resistance.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display probabilities
            if 'probability_class_0' in result and 'probability_class_1' in result:
                st.markdown("### 📊 Prediction Confidence")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Low MAR Probability", f"{result['probability_class_0']:.2%}")
                with col2:
                    st.metric("High MAR Probability", f"{result['probability_class_1']:.2%}")
                
                # Create probability bar chart
                prob_df = pd.DataFrame({
                    'Class': ['Low MAR', 'High MAR'],
                    'Probability': [result['probability_class_0'], result['probability_class_1']]
                })
                
                fig = px.bar(
                    prob_df,
                    x='Class',
                    y='Probability',
                    color='Class',
                    color_discrete_map={'Low MAR': '#28a745', 'High MAR': '#ffc107'},
                    title='Prediction Probabilities'
                )
                fig.update_layout(showlegend=False, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")


def batch_prediction_tab(deployment):
    """Tab for batch prediction from CSV"""
    st.subheader("📁 Batch Prediction from CSV")
    
    st.markdown("""
    Upload a CSV file containing multiple isolates for batch prediction.
    The file must include all required feature columns.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with antibiotic resistance data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully: {len(df)} samples")
            
            # Show preview
            with st.expander("👀 Data Preview (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Prediction options
            col1, col2 = st.columns(2)
            with col1:
                include_proba = st.checkbox("Include prediction probabilities", value=True)
            with col2:
                include_original = st.checkbox("Include original columns", value=True)
            
            # Predict button
            if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
                try:
                    with st.spinner("Making predictions..."):
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
                        
                        # Add probabilities
                        if probabilities is not None:
                            if probabilities.shape[1] == 2:
                                results[f'{task_name}_probability_class_0'] = probabilities[:, 0]
                                results[f'{task_name}_probability_class_1'] = probabilities[:, 1]
                    
                    st.success(f"✅ Predictions completed for {len(results)} samples")
                    
                    # Display summary
                    st.subheader("📊 Prediction Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(results))
                    with col2:
                        high_mar_count = (predictions == 1).sum()
                        st.metric("High MAR Predicted", high_mar_count)
                    with col3:
                        high_mar_pct = (high_mar_count / len(results)) * 100
                        st.metric("High MAR Percentage", f"{high_mar_pct:.1f}%")
                    
                    # Visualization
                    pred_counts = pd.Series(predictions).value_counts().reset_index()
                    pred_counts.columns = ['Prediction', 'Count']
                    pred_counts['Prediction'] = pred_counts['Prediction'].map({0: 'Low MAR', 1: 'High MAR'})
                    
                    fig = px.pie(
                        pred_counts,
                        values='Count',
                        names='Prediction',
                        title='Prediction Distribution',
                        color='Prediction',
                        color_discrete_map={'Low MAR': '#28a745', 'High MAR': '#ffc107'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results
                    with st.expander("📋 View All Results"):
                        st.dataframe(results, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except ValueError as e:
                    st.error(f"❌ Feature validation error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧬 AMR Model Deployment Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Predict Antimicrobial Resistance patterns using machine learning
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/favicon.png", width=50)
        st.title("⚙️ Configuration")
        
        # Model selection
        available_models = find_available_models()
        
        if not available_models:
            st.error("❌ No model files found in the current directory")
            st.info("Please ensure model .pkl files are available")
            st.stop()
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose a trained model for predictions"
        )
        
        # Load model
        try:
            with st.spinner("Loading model..."):
                deployment = load_model_cached(selected_model)
            st.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.stop()
        
        # Navigation
        st.markdown("---")
        st.markdown("### 📌 Navigation")
        
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Model Info", "🔬 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        display_model_info(deployment)
    
    with tab2:
        single_prediction_tab(deployment)
    
    with tab3:
        batch_prediction_tab(deployment)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        AMR Analysis Thesis Project | December 2025<br>
        Built with Streamlit 🎈
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
