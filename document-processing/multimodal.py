"""
Standalone script to extract entities from a document using Controlled Generation with Gemini 1.5
"""
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

vertexai.init(project="dialogflow-dan", location="us-central1")

PROMPT = """
You are a document entity extraction specialist.
Given a document, your task is to extract the text value of entities.
- Generate null for missing entities.
"""

# Load local PDF
with open("sample-documents/4.pdf", "rb") as pdf_file:
    pdf = pdf_file.read()

document = Part.from_data(data=pdf, mime_type="application/pdf")

# Load local PNG
# with open("sample-documents/4.png", "rb") as png_file:
#    png = png_file.read()
#
# document = Part.from_data(data=png, mime_type="image/png")

# Load document from Cloud Storage
# document = Part.from_uri(
#    mime_type="application/pdf",
#    uri="gs://doit-llm/generative-ai/pdf/4.pdf"
# )

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
                "required": [
                    "description",
                    "quantity",
                    "price",
                    "total",
                ],
            },
        }
    },
    "required": ["invoice_number"],
}

model = GenerativeModel("gemini-1.5-flash")

generation_config = GenerationConfig(
    max_output_tokens=8192,
    temperature=0,
    top_p=0.95,
    response_mime_type="application/json",
    response_schema=RESPONSE_SCHEMA,
)

responses = model.generate_content(
    [document, PROMPT],
    generation_config=generation_config,
)

print(responses.usage_metadata)

json_response = responses.candidates[0].content.parts[0].text

json_data = json.loads(json_response)
formatted_json = json.dumps(json_data, indent=4)

print(formatted_json)

with open("result.json", "w", encoding="utf-8") as json_file:

    json_file.write(formatted_json)

print("JSON output written to result.json")
