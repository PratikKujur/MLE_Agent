import streamlit as st
import os

# Lazy import to handle missing dependencies gracefully
get_eda = None
import_error = None

try:
    from main import get_eda
except ImportError as e:
    import_error = str(e)


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("EDA_Agent 🤖🧠🇦🇮👾")
st.header("Upload your dataset for EDA in .csv format")

# Check if dependencies are available
if import_error:
    st.error(f"⚠️ Missing dependencies: {import_error}")
    st.info("Make sure all required packages are installed using: pip install -r requirements.txt")
    st.stop()

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
        
        st.info(f"Processing {uploaded_file.name}...")
        
        # Call get_eda directly (returns dict)
        try:
            with st.spinner("🔄 Running EDA analysis..."):
                result = get_eda(file_path=file_path)
            
            # Check for errors in the response
            if isinstance(result, dict) and result.get("error"):
                st.error(f"Analysis Error: {result['error']}")
                st.session_state.error_message = result['error']
            else:
                st.success("✅ EDA Analysis Complete!")
                st.session_state.analysis_result = result
                st.session_state.error_message = None
        
        except Exception as e:
            st.session_state.error_message = f"Unexpected error: {str(e)}"
            st.session_state.analysis_result = None
            st.error(st.session_state.error_message)
            import traceback
            with st.expander("📋 Error Details"):
                st.code(traceback.format_exc())

# Display results
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
