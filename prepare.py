import os
import argparse
import pandas as pd
from datasets import load_dataset

def prepare_data(data_dir="data", output_dir="processed_data"):
    """
    Preparing the PUBHEALTH dataset for training by extracting relevant fields and saving it as a CSV file.

    Parameters:
    - data_dir (str): The directory where the raw dataset files are located. Defaults to 'data'.
    - output_dir (str): The directory where the processed dataset will be saved. Defaults to 'processed_data'.
    """
    # Ensuring the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Defining the path for the output file
    output_file = os.path.join(output_dir, "pubhealth_train.csv")

    # Loading the PUBHEALTH dataset splits
    print("Loading the PUBHEALTH dataset...")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "pubhealth_train.jsonl"),
        "validation": os.path.join(data_dir, "pubhealth_validation.jsonl"),
        "test": os.path.join(data_dir, "pubhealth_test.jsonl")
    })

    # Combining the train, validation, and test splits into a single dataframe
    combined_data = []
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split...")
        split_df = pd.DataFrame(split_data)
        split_df = split_df[["claim", "label", "explanation"]]
        combined_data.append(split_df)

    # Concatenating all splits into a single DataFrame
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Saving the processed data as a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")

def main():
    # Setting up argument parsing
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

    # Preparing the dataset and save it to the specified directory
    prepare_data(data_dir=args.data_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
