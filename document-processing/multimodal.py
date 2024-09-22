#!/usr/bin/env python
"""
Standalone script to extract entities from documents using Controlled Generation with Gemini 1.5.
Uses https://pypi.org/project/fsspec/ project to allow PDF reading documents from various sources,
including local, S3, GCS, HTTP(S). E.g:

python multimodal.py --input github://dzivkovi:gen-ai-livestream@main/document-processing/sample-documents/4.pdf
python multimodal.py --input https://raw.githubusercontent.com/dzivkovi/gen-ai-livestream/main/document-processing/sample-documents/4.pdf
python multimodal.py --output r.json --input file://sample-documents/4.pdf
python multimodal.py --output r.json --input ./sample-documents/4.pdf
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import argparse
import requests
import fsspec
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from dotenv import load_dotenv

# Configure logging
load_dotenv()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL)

# Constants
PROMPT = """
You are a document entity extraction specialist.
Given a document, your task is to extract the text value of entities.
- Generate null for missing entities.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "string"},
                    "price": {"type": "string"},
                    "total": {"type": "string"},
                },
                "required": ["description", "quantity", "price", "total"],
            },
        }
    },
    "required": ["invoice_number"],
}


def init_vertexai(project: str, location: str) -> None:
    """Initialize Vertex AI with the given project and location."""
    vertexai.init(project=project, location=location)


def read_document(file_path: str) -> bytes:
    """Read document from various sources using fsspec or requests."""
    if file_path.startswith(('http://', 'https://')):
        response = requests.get(file_path, timeout=60)
        response.raise_for_status()
        return response.content
    else:
        with fsspec.open(file_path, "rb") as file:
            return file.read()


def get_mime_type(file_path: str) -> str:
    """Determine MIME type based on file extension."""
    if file_path.endswith('.pdf'):
        return "application/pdf"
    elif file_path.endswith('.png'):
        return "image/png"
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def generate_content(model: GenerativeModel, document: Part, generation_config: GenerationConfig) -> Dict[str, Any]:
    """Generate content using the Gemini model."""
    responses = model.generate_content(
        [document, PROMPT],
        generation_config=generation_config,
    )
    logging.info("Generation metadata: %s", responses.usage_metadata)
    return json.loads(responses.candidates[0].content.parts[0].text)


def save_json(data: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """Save JSON data to a file or print to stdout."""
    if output_file:
        with fsspec.open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        logging.info("JSON output written to %s", output_file)
    else:
        json.dump(data, sys.stdout, indent=4)


def main():
    """
    Extract entities from documents using Gemini 1.5 Controlled Generation.
    """
    parser = argparse.ArgumentParser(description="Extract entities from documents using Gemini 1.5")
    parser.add_argument("--input", required=True, help="Input document path (local, S3, GCS, or HTTP(S))")
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    parser.add_argument("--project", default="dialogflow-dan", help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Google Cloud location")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model name")
    args = parser.parse_args()

    try:
        init_vertexai(args.project, args.location)

        document_data = read_document(args.input)
        mime_type = get_mime_type(args.input)
        document = Part.from_data(data=document_data, mime_type=mime_type)

        model = GenerativeModel(args.model)
        generation_config = GenerationConfig(
            max_output_tokens=8192,
            temperature=0,
            top_p=0.95,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )

        result = generate_content(model, document, generation_config)
        save_json(result, args.output)

    except Exception as e:
        logging.exception("An error occurred during execution: %s", e)
        raise


if __name__ == "__main__":
    main()
