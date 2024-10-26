# Dockerfile

# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY serve.py .

# Copy the model directory (fine-tuned model files)
COPY fine-tuned-model ./fine-tuned-model

# Expose the port on which the app will run
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
