"""
Streamlit-based front-end for our Data Extraction API.
"""
import os
import logging
import base64
import requests
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL)

# Define the API endpoint for document extraction
API_ENDPOINT = "http://127.0.0.1:8080/process_pdf"
API_ENDPOINT = os.getenv("API_ENDPOINT", "API_ENDPOINT must be set")


def extract_data(file, mimetype):
    """
    Function to extract text from PDF or image using the extraction API
    """
    file.seek(0)  # Ensure the file pointer is at the start
    files = {'file': (file.name, file, mimetype)}
    response = requests.post(API_ENDPOINT, files=files, timeout=60)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to extract data: {response.status_code}")
        return None


def display_pdf(file, height=400):
    """
    Function to display PDF in Streamlit using an iframe
    """
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    # pylint: disable=line-too-long
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="100%" height="{height}px" style="border: none;"></iframe>'
    )
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_image(file):
    """
    Function to display an image in Streamlit
    """
    image = Image.open(file)
    st.image(image, use_column_width=True)


# Streamlit UI
st.set_page_config(layout="wide")
st.title("Document Extraction with AI")
st.write("Upload a PDF or Image file to extract structured information.")

# Create a two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    # File uploader widget for PDF and image files
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        MIME_TYPE = "application/pdf" if uploaded_file.type == "application/pdf" else "image/jpeg"

        if st.button("Extract Information"):
            with st.spinner("Extracting information..."):
                extracted_data = extract_data(uploaded_file, MIME_TYPE)
                if extracted_data:
                    st.success("Information extracted successfully!")
                    st.write("## Extracted Information")
                    st.json(extracted_data)

with col2:
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.write("## PDF Preview")
            display_pdf(uploaded_file, height=400)
        else:
            st.write("## Image Preview")
            display_image(uploaded_file)
