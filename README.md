# Fact-Checking-Complete-MLOps-HuggingFace-Models
Task of verifying the veracity of claims using hugging face models and kuberbetes deployment

## Project Overview

The Fact-Checking Model pipeline includes the following key components:

1. **Data Ingestion** (`ingest.py`): Downloads and loads data for training.
2. **Data Preparation** (`prepare.py`): Cleans and preprocesses the data for fine-tuning.
3. **Model Training** (`train.py`): Fine-tunes a pretrained Hugging Face model on healthcare claim data.
4. **Model Evaluation** (`evaluate.py`): Assesses model performance using metrics like accuracy, F1-score, precision, and recall.
5. **Deployment** (`serve.py` and `service.yaml`): Deploys the model as a REST API with FastAPI and Kubernetes for scalable serving.


## Repository Structure

```plaintext
├── ingest.py          # Data ingestion script
├── prepare.py         # Data preprocessing script
├── train.py           # Model fine-tuning script
├── evaluate.py        # Model evaluation script
├── serve.py           # FastAPI app for serving predictions
├── service.yaml       # Kubernetes configuration file
├── README.md          # Project documentation
└── requirements.txt   # Required Python libraries
```

## Contributing

Contributions to enhance the functionality, add new models, or improve deployment strategies are welcome! Please fork this repository, make your changes, and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Hugging Face: [Transformers](https://github.com/huggingface/transformers)
- Kubernetes: [Kubernetes Docs](https://kubernetes.io/docs/)
- FastAPI: [FastAPI Framework](https://fastapi.tiangolo.com/)
