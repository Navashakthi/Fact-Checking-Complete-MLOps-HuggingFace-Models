import os
import argparse
from datasets import load_dataset

def download_data(save_dir="data"):
    """
    Downloads the PUBHEALTH dataset from Hugging Face and saves it locally.

    Parameters:
    - save_dir (str): The directory where the dataset will be saved. Defaults to 'data'.
    """
    # Load the PUBHEALTH dataset
    print("Loading the PUBHEALTH dataset...")
    dataset = load_dataset("ImperialCollegeLondon/health_fact")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the dataset to the specified directory
    print(f"Saving the dataset to {save_dir}...")
    for split in dataset.keys():
        file_path = os.path.join(save_dir, f"pubhealth_{split}.jsonl")
        dataset[split].to_json(file_path)
        print(f"Saved {split} split to {file_path}")

    print("Download and save complete.")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Download and save the PUBHEALTH dataset.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data",
        help="The directory where the dataset will be saved. Defaults to 'data'."
    )
    args = parser.parse_args()

    # Download the dataset and save it to the specified directory
    download_data(save_dir=args.save_dir)

if __name__ == "__main__":
    main()
