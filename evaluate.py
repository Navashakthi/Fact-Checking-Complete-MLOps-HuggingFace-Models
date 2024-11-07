# evaluate.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_metric, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the fine-tuned model and tokenizer
model_path = '/content/fine-tuned-model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load and preprocess the test data
pubhealth_df = pd.read_csv('/content/processed_data/pubhealth_train.csv')

def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        if isinstance(text, float):
            continue
        if not isinstance(text, str):
            text = str(text)
        cleaned_text = text.replace('\xa0', ' ').strip()
        cleaned_list.append(cleaned_text)
    return cleaned_list

pubhealth_df['cleaned_claim'] = clean_text(pubhealth_df['cleaned_claim'].tolist())
pubhealth_df.dropna(subset=['cleaned_claim', 'label'], inplace=True)

# Split into train and test datasets
_, test_df = train_test_split(pubhealth_df, test_size=0.2, random_state=42)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the dataset
def tokenize_function(texts):
    return tokenizer(texts['cleaned_claim'], padding="max_length", truncation=True)

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Evaluation function
def evaluate_model(model, dataset):
    metric = load_metric("accuracy")
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch in dataset:
            inputs = {key: batch[key].unsqueeze(0) for key in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class)
            labels.append(batch['label'].item())
    
    accuracy = metric.compute(predictions=predictions, references=labels)['accuracy']
    return accuracy

# Calculate and print accuracy
accuracy = evaluate_model(model, test_dataset)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
