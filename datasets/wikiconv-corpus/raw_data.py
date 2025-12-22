#!/usr/bin/env python3
"""
Downloads raw wikiconv datasets, and can also search for specific strings.
Parallelized for faster processing.
"""

import requests
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

# Global lock for thread-safe printing and counter for matches
print_lock = Lock()
matches_found = {'count': 0, 'files': []}

def get_file_list(figshare_article_id):
    """Fetch the list of ALL files from Figshare API (handles pagination)."""
    base_url = f"https://api.figshare.com/v2/articles/{figshare_article_id}/files"
    all_files = []
    page = 1
    page_size = 100  # Max allowed by Figshare API
    
    try:
        while True:
            params = {
                'page': page,
                'page_size': page_size
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            files = response.json()
            
            if not files:
                break
            
            all_files.extend(files)
            print(f"  Fetched page {page}: {len(files)} files (total so far: {len(all_files)})")
            
            if len(files) < page_size:
                # Last page
                break
            
            page += 1
        
        return all_files
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list: {e}")
        sys.exit(1)

def download_and_check_file(file_info, search_string, download_dir, idx, total):
    """
    Download a file, check for search string while streaming, and handle accordingly.
    Returns (found, file_name) tuple.
    """
    file_name = file_info['name']
    file_url = file_info['download_url']
    file_path = os.path.join(download_dir, file_name)
    
    with print_lock:
        print(f"[{idx}/{total}] Downloading: {file_name} ({file_info['size'] / (1024*1024):.2f} MB)...")
    
    try:
        # Download file with streaming
        response = requests.get(file_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Search while downloading (more efficient for large files)
        found = False
        chunk_size = 8192
        buffer = b''
        search_bytes = search_string.encode('utf-8')
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                
                # Search in overlapping buffer to catch strings across chunk boundaries
                buffer += chunk
                if search_bytes in buffer:
                    found = True
                    # Continue downloading but we know we found it
                
                # Keep last part of buffer for overlap check
                if len(buffer) > len(search_bytes) * 2:
                    buffer = buffer[-(len(search_bytes) * 2):]
        
        if found:
            with print_lock:
                print(f"  ✓ FOUND '{search_string}' in {file_name}!")
                print(f"  File saved at: {file_path}")
                matches_found['count'] += 1
                matches_found['files'].append(file_name)
            return (True, file_name)
        else:
            with print_lock:
                print(f"  String not found in {file_name}. Deleting...")
            os.remove(file_path)
            return (False, None)
        
    except requests.exceptions.RequestException as e:
        with print_lock:
            print(f"  Error downloading {file_name}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return (False, None)
    except Exception as e:
        with print_lock:
            print(f"  Error processing {file_name}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return (False, None)

def main():
    FIGSHARE_ARTICLE_ID = "7376003"  # english dataset, change for other languages
    SEARCH_STRING = "2052702.7345.7345"
    DOWNLOAD_DIR = "./wikiconv_downloads"
    MAX_WORKERS = 10  # Adjust based on your server's bandwidth and CPU
    
    print("=" * 60)
    print("WikiConv File Finder (Parallel - Keep All Matches)")
    print("=" * 60)
    print(f"Search string: '{SEARCH_STRING}'")
    print(f"Download directory: {DOWNLOAD_DIR}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print()
    
    # Create download directory
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get file list
    print("Fetching file list from Figshare...")
    files = get_file_list(FIGSHARE_ARTICLE_ID)
    
    if not files:
        print("No files found!")
        sys.exit(1)
    
    print(f"Found {len(files)} files.")
    print()
    
    start_time = time.time()
    
    # Process files in parallel
    START_INDEX = 1  # 1-based index, meaning skip first 88
    if START_INDEX > len(files):
        print(f"Start index ({START_INDEX}) is beyond available files ({len(files)}). Exiting.")
        sys.exit(1)

    files_to_process = files[START_INDEX - 1:]  # slice from the 89th file onward
    total_files = len(files_to_process)
    print(f"Processing files {START_INDEX}–{len(files)} ({total_files} total)...\n")

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks for remaining files only
        future_to_file = {
            executor.submit(
                download_and_check_file,
                file_info,
                SEARCH_STRING,
                DOWNLOAD_DIR,
                idx + START_INDEX - 1, 
                len(files)
            ): file_info
            for idx, file_info in enumerate(files_to_process, 1)
        }
        
        # process completed tasks
        for future in as_completed(future_to_file):
            found, file_name = future.result()
            completed += 1
            
            if completed % 50 == 0:
                with print_lock:
                    print(f"\n--- Progress: {completed}/{len(files)} files processed, {matches_found['count']} matches found ---\n")
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"COMPLETED: Processed all {len(files)} files.")
    print(f"Matches found: {matches_found['count']}")
    if matches_found['files']:
        print(f"\nFiles containing '{SEARCH_STRING}':")
        for match_file in matches_found['files']:
            print(f"  - {match_file}")
    else:
        print(f"\nSearch string '{SEARCH_STRING}' was NOT found in any file.")
    print(f"\nTime elapsed: {elapsed:.2f} seconds")
    print(f"Average: {elapsed/len(files):.2f} seconds per file")
    print("=" * 60)

if __name__ == "__main__":
    main()