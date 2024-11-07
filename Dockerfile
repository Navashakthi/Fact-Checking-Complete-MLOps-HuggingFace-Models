# Use a base image with Python
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn torch transformers gradio

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7863

# Start the app
CMD ["python", "main.py"]
