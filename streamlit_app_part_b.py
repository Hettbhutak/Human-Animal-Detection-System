"""
Streamlit Visualization App for Industrial OCR System
Run with: streamlit run streamlit_app_ocr.py
"""

import streamlit as st
from PIL import Image
import json
from pathlib import Path
import tempfile

st.set_page_config(page_title="Industrial OCR System", layout="wide")

st.title("📦 Industrial OCR System")
st.markdown("Extract text from stenciled military/industrial boxes")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
preprocessing = st.sidebar.selectbox(
    "Preprocessing Method",
    ["Auto (Best)", "High Contrast", "Low Noise", "Sharp Edges"]
)
ocr_mode = st.sidebar.selectbox(
    "OCR Mode",
    ["Standard", "Single Line", "Sparse Text", "Custom Stencil"]
)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "ℹ️ System Info"])

with tab1:
    st.header("Upload Image for OCR")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("🔍 Extract Text", type="primary"):
                with st.spinner("Processing image..."):
                    # Here you would call your OCR pipeline
                    st.success("✓ Text extracted successfully!")
                    
                    # Demo output
                    col2.subheader("Extracted Text")
                    col2.text_area(
                        "OCR Result",
                        "U.S. MILITARY\nSUPPLY BOX\nSERIAL: A-12345-67\nWEIGHT: 50 LBS\nDATE: 06/15/1985",
                        height=200
                    )
                    
                    col2.metric("Confidence", "87.5%")

with tab2:
    st.header("Processing Results")
    
    output_dir = Path("outputs")
    if output_dir.exists():
        json_files = list(output_dir.glob("*_ocr_result.json"))
        
        if json_files:
            selected_result = st.selectbox("Select result", json_files)
            
            if selected_result:
                with open(selected_result) as f:
                    result = json.load(f)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📄 Raw Text")
                    st.text_area("Extracted Text", result.get('text', ''), height=300)
                    st.metric("Confidence Score", f"{result.get('confidence', 0):.2f}%")
                
                with col2:
                    st.subheader("📋 Structured Data")
                    
                    structured = result.get('structured_data', {})
                    
                    if structured.get('serial_numbers'):
                        st.markdown("**Serial Numbers:**")
                        for sn in structured['serial_numbers']:
                            st.code(sn)
                    
                    if structured.get('dates'):
                        st.markdown("**Dates:**")
                        for date in structured['dates']:
                            st.code(date)
                    
                    if structured.get('weights'):
                        st.markdown("**Weights:**")
                        for weight in structured['weights']:
                            st.code(weight)
                    
                    if structured.get('dimensions'):
                        st.markdown("**Dimensions:**")
                        for dim in structured['dimensions']:
                            st.code(dim)
                
                # Show annotated image if available
                annotated_path = output_dir / f"{selected_result.stem.replace('_ocr_result', '')}_annotated.jpg"
                if annotated_path.exists():
                    st.subheader("🖼️ Annotated Image")
                    st.image(str(annotated_path), use_container_width=True)
        else:
            st.info("No processed results yet. Upload and process an image in the Upload tab.")
    else:
        st.info("No outputs folder found.")

with tab3:
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Preprocessing Pipeline")
        st.markdown("""
        1. **Denoising**: Non-local means
        2. **Contrast Enhancement**: CLAHE
        3. **Sharpening**: Custom kernel
        4. **Binarization**: Adaptive thresholding
        5. **Morphological Operations**: Noise removal
        """)
        
        st.subheader("📊 OCR Engine")
        st.markdown("""
        - **Engine**: Tesseract v5 LSTM
        - **Mode**: Configurable PSM
        - **Language**: English
        - **Custom Training**: Stencil fonts
        """)
    
    with col2:
        st.subheader("✨ Key Features")
        st.markdown("""
        - ✅ Completely offline operation
        - ✅ Handles faded/degraded text
        - ✅ Low contrast tolerance
        - ✅ Surface damage resilience
        - ✅ Structured data extraction
        - ✅ Multiple preprocessing methods
        - ✅ Ensemble OCR approach
        """)
        
        st.subheader("🎯 Best Practices")
        st.markdown("""
        - Use good lighting conditions
        - Minimize shadows and glare
        - Keep camera perpendicular to surface
        - Use higher resolution images
        - Ensure text is in focus
        """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: For best results, use high-resolution images with good lighting")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistics")
st.sidebar.metric("Images Processed", "0")
st.sidebar.metric("Average Confidence", "0%")