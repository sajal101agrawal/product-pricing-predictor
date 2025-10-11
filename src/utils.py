import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def download_image_with_timeout(image_link, savefolder, timeout=10):
    """Download a single image with timeout and better error handling."""
    if not isinstance(image_link, str) or not image_link.strip():
        return False, "Invalid URL"
    
    try:
        filename = Path(image_link).name
        if not filename or filename == '':
            # Generate a filename from URL if extraction fails
            filename = f"image_{hash(image_link)}.jpg"
        
        image_save_path = os.path.join(savefolder, filename)
        
        if os.path.exists(image_save_path):
            return True, "Already exists"
        
        # Use requests with timeout instead of urllib
        response = requests.get(image_link, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Save the image
        with open(image_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True, "Downloaded"
        
    except requests.exceptions.Timeout:
        return False, f"Timeout after {timeout}s"
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def download_image(image_link, savefolder):
    """Legacy function for compatibility."""
    success, _ = download_image_with_timeout(image_link, savefolder)
    return success

def download_images(image_links, download_folder, max_workers=20, timeout=10):
    """Download images using ThreadPoolExecutor for better stability."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder, exist_ok=True)
    
    # Filter out None and empty links
    valid_links = [link for link in image_links if link and isinstance(link, str) and link.strip()]
    
    print(f"Found {len(valid_links)} valid links out of {len(image_links)} total")
    
    successful = 0
    failed = 0
    errors = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_image_with_timeout, url, download_folder, timeout): url 
            for url in valid_links
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(valid_links), desc="Downloading images") as pbar:
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success, message = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        errors[url] = message
                except Exception as exc:
                    failed += 1
                    errors[url] = f"Exception: {exc}"
                
                pbar.update(1)
                pbar.set_postfix({'success': successful, 'failed': failed})
    
    # Print summary
    print(f"\nDownload complete:")
    print(f"  Successfully downloaded: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0 and len(errors) > 0:
        print(f"\nShowing first 5 errors:")
        for i, (url, error) in enumerate(list(errors.items())[:5]):
            print(f"  - {url}: {error}")
        
        if failed > 5:
            print(f"  ... and {failed - 5} more errors")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(download_folder)
        free_gb = free / (1024**3)
        print(f"\nDisk space after download: {free_gb:.1f} GB available")
    except:
        pass
    
    return successful, failed