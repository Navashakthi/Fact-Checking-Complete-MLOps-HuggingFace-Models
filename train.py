# Import necessary libraries
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


pubhealth_df = pd.read_csv('/content/processed_data/pubhealth_train.csv')

def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        # Check if the text is a string, if not, convert it to string
        if isinstance(text, float):
            # Skip NaN or None values (optional: replace with a placeholder)
            continue
        if not isinstance(text, str):
            text = str(text)
        cleaned_text = text.replace('\xa0', ' ').strip()
        cleaned_list.append(cleaned_text)
    return cleaned_list

def apply_clean_text(df):
    # Drop rows with NaN values in the 'cleaned_claim' column
    df_cleaned = df.dropna(subset=['cleaned_claim']).copy()

    # Apply the clean_text function to the 'cleaned_claim' column using list comprehension
    df_cleaned['cleaned_claim'] = clean_text(df_cleaned['cleaned_claim'].tolist())

    # Replace the original dataframe's cleaned_claim column with the cleaned data while maintaining the original indices
    df.update(df_cleaned)

    # Drop rows where the 'cleaned_claim' column might still be empty or contain NaN
    df = df.dropna(subset=['cleaned_claim'])

    return df

df = apply_clean_text(pubhealth_df)
X = list(df['cleaned_claim'])
y = list(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# Check for any -1 labels and remove them if necessary
train_df = train_df[train_df['label'] >= 0]
val_df = val_df[val_df['label'] >= 0]

# Convert the training and validation dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("austinmw/distilbert-base-uncased-finetuned-health_facts")
model = AutoModelForSequenceClassification.from_pretrained("austinmw/distilbert-base-uncased-finetuned-health_facts")

# Tokenize the dataset
def tokenize_function(texts):
    return tokenizer(texts['cleaned_claim'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set the format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')
