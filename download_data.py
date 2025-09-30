"""
Download dataset files from Google Drive
Run this script before running the recommendation system
"""

import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Get download confirmation token"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save downloaded content to file with progress bar"""
    CHUNK_SIZE = 32768
    
    # Get file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    """Download all required dataset files"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("=" * 60)
    print("Downloading Steam Game Recommendations Dataset")
    print("=" * 60)
    
    # File IDs from Google Drive
    files = {
        'users.csv': '1loW5XkftMqZNbeoDFlAs9uQODkzYqLui',
        'recommendations.csv': '1AQGA9fiK8XvJaFfyBO-jrynNZ2NpRsoJ'
    }
    
    for filename, file_id in files.items():
        destination = os.path.join('data', filename)
        
        if os.path.exists(destination):
            print(f"\n✓ {filename} already exists, skipping...")
            continue
        
        print(f"\nDownloading {filename}...")
        try:
            download_file_from_google_drive(file_id, destination)
            print(f"✓ {filename} downloaded successfully!")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # Check games.csv
    if os.path.exists('data/games.csv'):
        print("\n✓ All data files are ready!")
    else:
        print("\n⚠ Warning: games.csv not found in data/ directory")

if __name__ == "__main__":
    main()