
import streamlit as st
import requests
from PIL import Image
import io
from datetime import datetime
import pandas as pd
import os

# Configuration - Use environment variable or default to localhost for local dev
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000") + "/api/edgelens"

# Page configuration
st.set_page_config(
    page_title="EdgeLens - Defect Detection",
    page_icon="🔍",
    layout="wide"
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .defective {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .ok {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 EdgeLens Defect Detection System</h1>', unsafe_allow_html=True)

# Health check at top
api_url = API_BASE_URL
col_status1, col_status2 = st.columns([3, 1])
with col_status2:
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Error")
    except Exception as e:
        st.error(f"❌ Backend Offline")

st.divider()

# Main content area - Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Detect Defects", "📜 History", "🔄 Retraining"])

# Tab 1: Image Upload and Prediction
with tab1:
    st.header("Upload Image for Defect Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image of the product to check for defects"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Analyze for Defects", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Prepare the file for API request
                        uploaded_file.seek(0)  # Reset file pointer
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        
                        # Make prediction request
                        response = requests.post(f"{api_url}/predict", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            with col2:
                                st.subheader("📊 Analysis Results")
                                
                                # Determine prediction class
                                prediction = result.get("label", "Unknown")
                                confidence = result.get("confidence", 0.0)
                                inference_time = result.get("inference_time", 0) * 1000  # Convert to ms
                                
                                # Display prediction with color coding
                                if prediction.lower() == "defective":
                                    st.markdown(
                                        f'<div class="prediction-box defective">'
                                        f'<h2>⚠️ DEFECTIVE</h2>'
                                        f'<p><strong>Confidence:</strong> {confidence:.2%}</p>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="prediction-box ok">'
                                        f'<h2>✅ OK</h2>'
                                        f'<p><strong>Confidence:</strong> {confidence:.2%}</p>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Additional metrics
                                st.metric("Inference Time", f"{inference_time:.2f} ms")
                                
                                # Show full response details
                                with st.expander("📋 View Full Response"):
                                    st.json(result)
                        else:
                            st.error(f"❌ Error: {response.status_code} - {response.text}")
                    
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. Please try again.")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Cannot connect to backend. Is the server running?")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Please upload an image to begin defect detection")

# Tab 2: Prediction History
with tab2:
    st.header("Prediction History")
    
    if st.button("🔄 Refresh History"):
        st.rerun()
    
    try:
        response = requests.get(f"{api_url}/history", timeout=10)
        
        if response.status_code == 200:
            history = response.json()
            
            if history and len(history) > 0:
                # Convert to DataFrame for better display
                df_data = []
                for idx, record in enumerate(history, 1):
                    df_data.append({
                        "#": idx,
                        "Timestamp": record.get("timestamp", "N/A"),
                        "Prediction": record.get("label", "Unknown"),
                        "Confidence": f"{record.get('confidence', 0):.2%}"
                    })
                
                df = pd.DataFrame(df_data)
                
                # Display as table
                st.dataframe(df, hide_index=True)
                
                # Statistics
                st.subheader("📊 Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total = len(history)
                    st.metric("Total Predictions", total)
                
                with col2:
                    defective_count = sum(1 for h in history if str(h.get("label", "")).lower() == "defective")
                    st.metric("Defective Count", defective_count)
                
                with col3:
                    ok_count = total - defective_count
                    st.metric("OK Count", ok_count)
                
                # Show raw data
                with st.expander("📋 View Raw Data"):
                    st.json(history)
            else:
                st.info("No prediction history available yet.")
        else:
            st.error(f"Failed to fetch history: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend. Is the server running?")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Tab 3: Model Retraining
with tab3:
    st.header("Model Retraining")
    
    st.info("""
    **Note:** This feature triggers an asynchronous retraining simulation.
    In production, this would initiate the model retraining pipeline with new data.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Trigger Retraining")
        st.write("Click the button below to initiate model retraining:")
        
        if st.button("🔄 Start Retraining", type="primary"):
            with st.spinner("Triggering retraining process..."):
                try:
                    response = requests.post(f"{api_url}/retrain", timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result.get('message', 'Retraining triggered successfully!')}")
                        
                        with st.expander("📋 View Response"):
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to backend. Is the server running?")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("ℹ️ Info")
        st.write("""
        Retraining Process:
        - Collects new training data
        - Validates data quality
        - Trains updated model
        - Deploys new version
        """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
    EdgeLens Defect Detection System | FastAPI + PyTorch + Streamlit
    </div>
""", unsafe_allow_html=True)
