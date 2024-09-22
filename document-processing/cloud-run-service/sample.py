"""
TOTO: Redundant code, use CURL instead
curl -X POST "http://127.0.0.1:8080/process_pdf" -F "file=@../sample-documents/4.pdf"

This script sends a PDF file to a Cloud Run service for processing,
but you can also use CURL:
"""
import requests

# The URL of your deployed Cloud Run service
API_ENDPOINT = "http://127.0.0.1:8080/process_pdf"

# The path to the PDF file you want to upload
PDF_FILE_PATH = "4.pdf"

# Send the POST request with the PDF file
with open(PDF_FILE_PATH, 'rb') as pdf_file:
    files = {'file': pdf_file}
    response = requests.post(API_ENDPOINT, files=files, timeout=60)

# Check if the request was successful
if response.status_code == 200:
    print("Response JSON:")
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
