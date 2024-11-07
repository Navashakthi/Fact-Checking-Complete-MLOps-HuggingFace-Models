from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def clean_text(text_list):
    """Clean text by removing unwanted characters and handling non-string values."""
    cleaned_list = []
    for text in text_list:
        if isinstance(text, float):
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
    model_path = '/content/fine-tuned-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    test_df = pd.read_csv('/content/processed_data/pubhealth_test.csv')
    test_df = apply_clean_text(test_df)
    test_df = test_df[test_df['label'].isin([0, 1, 2, 3])]  # Ensure labels are in range

    test_dataset = Dataset.from_pandas(test_df)
    def tokenize_function(examples):
        return tokenizer(examples['cleaned_claim'], padding="max_length", truncation=True, max_length=128)

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
