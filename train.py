import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def load_data(file_path):
    """
    Loads the prepared PUBHEALTH dataset from a CSV file and converts it into a Hugging Face Dataset.

    Parameters:
    - file_path (str): The path to the CSV file containing the dataset.

    Returns:
    - Dataset: A Hugging Face Dataset object.
    """
    # Load the CSV file using pandas
    df = pd.read_csv(file_path)

    # Map labels to numeric values (true, false, unproven, mixture -> 0, 1, 2, 3)
    label_mapping = {"true": 0, "false": 1, "unproven": 2, "mixture": 3}
    df['label'] = df['label'].map(label_mapping)

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def tokenize_function(examples):
    """
    Tokenizes the input examples using a pre-trained tokenizer.

    Parameters:
    - examples (dict): A dictionary containing input text fields.

    Returns:
    - dict: Tokenized examples with input IDs and attention masks.
    """
    return tokenizer(examples["claim"], truncation=True, padding="max_length", max_length=128)

def main(data_file, model_name="austinmw/distilbert-base-uncased-finetuned-health_facts", output_dir="model_output"):
    # Load the tokenizer and model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # Load and tokenize the dataset
    dataset = load_data(data_file)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Typically, you'd use a validation set here.
        tokenizer=tokenizer,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # Example usage: train_model.py --data_file processed_data/pubhealth_train.csv
    data_file = "processed_data/pubhealth_train.csv"
    main(data_file)
