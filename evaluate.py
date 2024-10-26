# evaluate.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_text(text_list):
    """Clean text by removing unwanted characters and handling non-string values."""
    cleaned_list = []
    for text in text_list:
        if isinstance(text, float):  # Skip NaN or None values
            continue
        if not isinstance(text, str):
            text = str(text)
        cleaned_text = text.replace('\xa0', ' ').strip()
        cleaned_list.append(cleaned_text)
    return cleaned_list

def apply_clean_text(df):
    """Apply text cleaning to the 'cleaned_claim' column."""
    df_cleaned = df.dropna(subset=['cleaned_claim']).copy()
    df_cleaned['cleaned_claim'] = clean_text(df_cleaned['cleaned_claim'].tolist())
    df.update(df_cleaned)
    df = df.dropna(subset=['cleaned_claim'])
    return df

def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1-score for evaluation."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Load the fine-tuned model and tokenizer
    model_path = './fine-tuned-model'  # Path to the saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load the test dataset (assuming a similar format as the training set)
    test_df = pd.read_csv('/content/processed_data/pubhealth_test.csv')
    
    # Apply text cleaning to the test dataset
    test_df = apply_clean_text(test_df)

    # Convert the test dataframe to a Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize the test dataset
    def tokenize_function(examples):
        return tokenizer(examples['cleaned_claim'], padding="max_length", truncation=True)

    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set the format to PyTorch tensors
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define training arguments (can reuse the ones from training)
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
