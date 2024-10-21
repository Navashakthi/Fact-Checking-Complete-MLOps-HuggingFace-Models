import os
import requests
import argparse

def download_data(url, save_folder, filename):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Define the file path
    file_path = os.path.join(save_folder, filename)
    
    try:
        # Send a GET request to download the data
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors
        
        # Write the content to a file
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Data successfully downloaded and saved to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download data and save it to a specified folder.")
    parser.add_argument("--url", type=str, required=True, help="URL of the data to download")
    parser.add_argument("--folder", type=str, default="data", help="Folder where the data should be saved (default: 'data')")
    parser.add_argument("--filename", type=str, default="datafile", help="Name of the file to save (default: 'datafile')")
    
    args = parser.parse_args()
    
    # Download the data
    download_data(args.url, args.folder, args.filename)

if __name__ == "__main__":
    main()
