"""Helper script to download the Gloomhaven rulebook PDF."""

import requests
from pathlib import Path
from src.config import Config

def download_rulebook():
    """Download the Gloomhaven rulebook PDF if it doesn't exist."""
    
    Config.ensure_directories()
    
    if Config.PDF_PATH.exists():
        print(f"✓ PDF already exists at {Config.PDF_PATH}")
        return
    
    url = "https://cdn.1j1ju.com/medias/8d/c5/21-gloomhaven-rulebook.pdf"
    
    print(f"Downloading rulebook from {url}...")
    print(f"This may take a few minutes...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(Config.PDF_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='')
        
        print(f"\n✓ Successfully downloaded rulebook to {Config.PDF_PATH}")
        print(f"Size: {Config.PDF_PATH.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error downloading rulebook: {e}")
        print("\nPlease download manually from:")
        print(url)
        print(f"And save it to: {Config.PDF_PATH}")

if __name__ == "__main__":
    download_rulebook()

