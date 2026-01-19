import streamlit as st
import requests
import os
import json

BASE_URL = "http://127.0.0.1:8000"

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None

# Streamlit app to interact with the FastAPI backend
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("EDA_Agent")
st.header("Upload your dataset for EDA in .csv format")

uploaded_file = st.file_uploader(
    label="Upload your dataset for EDA in .csv format",
    key="file_uploaded",
    type=["csv"]
)

if uploaded_file is not None:
    if st.button("Upload dataset", key="dataset_uploader_button"):
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"Uploading {uploaded_file.name}...")
        
        # Send to API
        try:
            response = requests.post(
                f"{BASE_URL}/upload",
                json={"file_path": file_path},
                timeout=300  # 5 minute timeout for long-running analysis
            )
            response.raise_for_status()
            result = response.json()
            
            # Debug: Print the response structure
            st.write("📋 Raw API Response:")
            st.json(result)
            
            # Check for errors in the response
            if result.get("error"):
                st.error(f"API Error: {result['error']}")
            
            st.session_state.analysis_result = result
            st.session_state.error_message = None
            st.rerun()
        
        except requests.exceptions.Timeout:
            st.session_state.error_message = "Request timed out. The analysis took too long."
            st.session_state.analysis_result = None
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.session_state.error_message = f"Cannot connect to API at {BASE_URL}. Make sure the FastAPI server is running on: uvicorn api:app --reload"
            st.session_state.analysis_result = None
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.session_state.error_message = f"Request error: {str(e)}"
            st.session_state.analysis_result = None
            st.rerun()
        except Exception as e:
            st.session_state.error_message = f"Unexpected error: {str(e)}"
            st.session_state.analysis_result = None
            st.rerun()

# Display results
if st.session_state.error_message:
    st.error(st.session_state.error_message)

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    st.success("✅ EDA Analysis Complete!")
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Domain Expert Analysis")
        if result.get("Domain_expert"):
            st.json(result["Domain_expert"])
        else:
            st.info("No domain expert analysis available")
    
    with col2:
        st.subheader("📈 Dataset Profile")
        if result.get("Dataset_profiler"):
            st.text(str(result["Dataset_profiler"])[:500])  # Show first 500 chars
        else:
            st.info("No dataset profile available")
    
    st.divider()
    
    st.subheader("🎯 EDA Strategy")
    if result.get("EDA_Resonner"):
        if isinstance(result["EDA_Resonner"], dict):
            st.json(result["EDA_Resonner"])
        else:
            st.text(str(result["EDA_Resonner"]))
    else:
        st.info("No EDA strategy available")
    
    st.divider()
    
    st.subheader("📋 EDA Report")
    if result.get("EDA_report_generator"):
        if isinstance(result["EDA_report_generator"], dict):
            st.json(result["EDA_report_generator"])
        else:
            st.text_area("Report", value=str(result["EDA_report_generator"]), height=300, disabled=True)
    else:
        st.info("No EDA report available")
    
    if st.button("Clear Results", key="clear_button"):
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        st.rerun()
