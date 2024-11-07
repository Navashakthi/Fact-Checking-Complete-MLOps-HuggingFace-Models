# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr
import threading

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer (ensure your model path or model loading code is correct)
model_path = '/content/fine-tuned-model'  # Adjust the model path if needed
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the Pydantic model for input validation
class Claim(BaseModel):
    text: str

# FastAPI endpoint to get prediction
@app.post("/claim/v1/predict")
async def predict_claim(claim: Claim):
    try:
        # Tokenize input text
        inputs = tokenizer(claim.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
        return {"claim": claim.text, "veracity": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the Gradio interface function
def gradio_predict(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
        return f"Label {predicted_label}"  # Returns the label directly as 0, 1, 2, or 3
    except Exception as e:
        return f"Error: {e}"

# Set up Gradio interface
gr_interface = gr.Interface(fn=gradio_predict, inputs="text", outputs="text",
                            title="Claim Veracity Predictor",
                            description="Enter a claim to predict its veracity.")

# Run Gradio in a separate thread
def run_gradio():
    gr_interface.launch(server_name="0.0.0.0", server_port=7864)

gradio_thread = threading.Thread(target=run_gradio)
gradio_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
