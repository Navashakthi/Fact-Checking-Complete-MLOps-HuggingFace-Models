# serve.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the input schema using Pydantic
class ClaimRequest(BaseModel):
    claim: str

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define a helper function to perform prediction
def predict_claim(claim_text: str):
    # Tokenize the input claim
    inputs = tokenizer(claim_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()
    return predictions

# Define the /claim/v1/predict endpoint
@app.post("/claim/v1/predict")
async def predict_label(request: ClaimRequest):
    try:
        # Get the claim from the request
        claim_text = request.claim
        
        # Get the prediction
        label = predict_claim(claim_text)
        
        # Return the predicted label as a response
        return {"label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
