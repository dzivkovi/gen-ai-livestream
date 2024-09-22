## Usage

replace `dialogflow-dan` with your project ID

````
gcloud builds submit --tag gcr.io/dialogflow-dan/document-understanding

gcloud run deploy --image gcr.io/dialogflow-dan/document-understanding --platform managed --allow-unauthenticated

````