# Fact-Checking Model: Complete MLOps Pipeline with Hugging Face Models

This repository implements a comprehensive MLOps pipeline for fact-checking claims using Hugging Face transformer models. The project includes data ingestion, preprocessing, model training, evaluation, and deployment as a FastAPI service. The deployment is set up to run in a Kubernetes environment, making it scalable and production-ready.

### Repository Overview

The key components of the pipeline are as follows:
- **`ingest.py`**: Downloads and loads the dataset.
- **`prepare.py`**: Preprocesses data for model training.
- **`train.py`**: Fine-tunes a pre-trained Hugging Face model for fact-checking.
- **`evaluate.py`**: Evaluates the trained model on validation data.
- **`serve.py`**: Deploys the model as a REST API with FastAPI.
- **`service.yaml`**: Configures deployment with Kubernetes.

### Getting Started

Follow these instructions to set up and run the project on your local environment.

#### 1. Clone the Repository

```bash
git clone https://github.com/Navashakthi/Fact-Checking-Complete-MLOps-using-HuggingFace-Models.git
cd Fact-Checking-Complete-MLOps-using-HuggingFace-Models
```

#### 2. Follow Execution.ipynb

Follow the implementation done in the notebook that includes the following steps;

- Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

#### 3. Data Ingestion

Download and load the dataset with `ingest.py`. This script fetches data for training and saves it in a specified directory.

```bash
python ingest.py
```

#### 4. Data Preparation

Clean and preprocess the dataset by running `prepare.py`. This step applies data cleaning and tokenization, preparing it for model training.

```bash
python prepare.py
```

#### 5. Model Training

Train the model using `train.py`, which fine-tunes a Hugging Face transformer model on the dataset. You can adjust model and training parameters in the script as needed.

```bash
python train.py
```

#### 6. Model Evaluation

Evaluate the trained model on a test set using `evaluate.py`, which will output metrics like accuracy, F1-score, precision, and recall.

```bash
python evaluate.py
```

#### 7. Serve the Model with FastAPI

To deploy the model as a REST API, run `serve.py` using Uvicorn. This will launch a FastAPI application locally.

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Access the API at `http://localhost:8000`.

#### 8. Deploy with Kubernetes

To deploy the FastAPI service on Kubernetes, apply the configuration in `service.yaml`. Ensure that Kubernetes is set up and that `kubectl` is configured correctly.

```bash
kubectl apply -f service.yaml
```

### Sample API Output
![Screenshot 2024-11-07 at 7 58 37 PM](https://github.com/user-attachments/assets/fadfc0e6-76d4-4ee3-993c-154a05c7f89a)


### Folder Structure

```
├── ingest.py             # Data ingestion script
├── prepare.py            # Data preprocessing script
├── train.py              # Model training and fine-tuning script
├── evaluate.py           # Model evaluation script
├── serve.py              # FastAPI service for serving predictions
├── service.yaml          # Kubernetes configuration file for deployment
├── requirements.txt      # Dependencies list
└── README.md             # Documentation
```

### Monitoring

To monitor and update the fact-checking model, we can track performance metrics like accuracy, F1-score, precision, and recall on live data, watching for any declines that could indicate the need for retraining. Monitoring for data drift and concept drift would help detect changes in data patterns or the relationship between claims and veracity. A feedback loop for flagged errors would also help identify problematic predictions, and the model could be scheduled for periodic retraining or updates if performance falls below defined thresholds.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Hugging Face: [Transformers](https://github.com/huggingface/transformers)
- Kubernetes: [Kubernetes Docs](https://kubernetes.io/docs/)
- FastAPI: [FastAPI Framework](https://fastapi.tiangolo.com/)
