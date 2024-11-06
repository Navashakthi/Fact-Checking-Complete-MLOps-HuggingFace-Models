# Import libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pyngrok import ngrok

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer (ensure your model path or model loading code is correct)
model_path = './fine-tuned-model'  # Adjust the model path if needed
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

class Claim(BaseModel):
    text: str

@app.post("/claim/v1/predict")
async def predict_claim(claim: Claim):
    try:
        inputs = tokenizer(claim.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
        return {"claim": claim.text, "veracity": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Set up ngrok
public_url = ngrok.connect(8000)
print("FastAPI public URL:", public_url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
