#!/usr/bin/env python
"""
Standalone script to extract entities from documents using Controlled Generation with Gemini 1.5.
Supports reading documents from various sources including local, S3, GCS, and HTTP(S).
"""

import argparse
import json
import logging
from typing import Dict, Any

import fsspec
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Read document from various sources using fsspec."""
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
    logger.info("Generation metadata: %s", responses.usage_metadata)
    return json.loads(responses.candidates[0].content.parts[0].text)


def save_json(data: Dict[str, Any], output_file: str) -> None:
    """Save JSON data to a file."""
    with fsspec.open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("JSON output written to %s", output_file)


def main():
    parser = argparse.ArgumentParser(description="Extract entities from documents using Gemini 1.5")
    parser.add_argument("--input", required=True, help="Input document path (local, S3, GCS, or HTTP(S))")
    parser.add_argument("--output", default="result.json", help="Output JSON file path")
    parser.add_argument("--project", default="dialogflow-dan", help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Google Cloud location")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum output tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    args = parser.parse_args()

    try:
        init_vertexai(args.project, args.location)

        document_data = read_document(args.input)
        mime_type = get_mime_type(args.input)
        document = Part.from_data(data=document_data, mime_type=mime_type)

        model = GenerativeModel(args.model)
        generation_config = GenerationConfig(
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )

        result = generate_content(model, document, generation_config)
        save_json(result, args.output)

    except Exception as e:
        logger.exception("An error occurred during execution:")
        raise

if __name__ == "__main__":
    main()
