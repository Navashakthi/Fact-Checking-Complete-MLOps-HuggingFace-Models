import os
import argparse
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

# Download NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocesses text data: lowers case, removes URLs, special characters, stopwords, and lemmatizes."""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

def clean_data(df):
    """Cleans the DataFrame by dropping NaN values and preprocessing text."""
    # Drop rows with missing values in any of the specified columns
    df = df.dropna(subset=['claim', 'explanation', 'label'])
    # Optionally, reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    # Apply text preprocessing to 'claim' and 'explanation' columns
    df['cleaned_claim'] = df['claim'].apply(preprocess_text)
    df['cleaned_explanation'] = df['explanation'].apply(preprocess_text)

    return df

def prepare_data(data_dir="data", output_dir="processed_data"):
    """
    Prepares the PUBHEALTH dataset for training by extracting relevant fields and saving it as a CSV file.

    Parameters:
    - data_dir (str): The directory where the raw dataset files are located. Defaults to 'data'.
    - output_dir (str): The directory where the processed dataset will be saved. Defaults to 'processed_data'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the path for the output file
    output_file = os.path.join(output_dir, "pubhealth_train.csv")

    # Load the PUBHEALTH dataset splits
    print("Loading the PUBHEALTH dataset...")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "pubhealth_train.jsonl"),
        "validation": os.path.join(data_dir, "pubhealth_validation.jsonl"),
        "test": os.path.join(data_dir, "pubhealth_test.jsonl")
    })

    # Combine the train, validation, and test splits into a single dataframe
    combined_data = []
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split...")
        split_df = pd.DataFrame(split_data)
        split_df = split_df[["claim", "label", "explanation"]]
        combined_data.append(split_df)

    # Concatenate all splits into a single DataFrame
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Clean the combined data
    cleaned_df = clean_data(combined_df)

    # Save the processed data as a CSV file
    cleaned_df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Prepare the PUBHEALTH dataset for model training.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="The directory where the raw dataset files are located. Defaults to 'data'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="The directory where the processed dataset will be saved. Defaults to 'processed_data'."
    )
    args = parser.parse_args()

    # Prepare the dataset and save it to the specified directory
    prepare_data(data_dir=args.data_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
