import os
import argparse
from datasets import load_dataset

def download_data(save_dir="data"):
    """
    Downloading the PUBHEALTH dataset from Hugging Face and saving it locally.

    Parameters:
    - save_dir (str): The directory where the dataset will be saved. Defaults to 'data'.
    """
    # Loading the PUBHEALTH dataset with trust_remote_code set to True
    print("Loading the PUBHEALTH dataset...")
    dataset = load_dataset("ImperialCollegeLondon/health_fact", trust_remote_code=True)

    # Ensuring the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Saving the dataset to the specified directory
    print(f"Saving the dataset to {save_dir}...")
    for split in dataset.keys():
        file_path = os.path.join(save_dir, f"pubhealth_{split}.jsonl")
        dataset[split].to_json(file_path)
        print(f"Saved {split} split to {file_path}")

    print("Download and save complete.")

def main():
    # Setting up argument parsing
    parser = argparse.ArgumentParser(description="Download and save the PUBHEALTH dataset.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data",
        help="The directory where the dataset will be saved. Defaults to 'data'."
    )
    args = parser.parse_args()

    # Downloading the dataset and save it to the specified directory
    download_data(save_dir=args.save_dir)

if __name__ == "__main__":
    main()
