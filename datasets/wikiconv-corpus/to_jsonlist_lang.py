#!/usr/bin/env python3
"""
Downloads raw wikiconv dataset based on specify id and converts them to JSONLIST format in parallel.
Deletes raw files after successful conversion.
"""

import requests
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
from collections import defaultdict
import time

print_lock = Lock()
stats = {
    'downloaded': 0,
    'converted': 0,
    'failed': 0,
    'total_conversations': 0
}

def get_file_list(figshare_article_id):
    """Fetch list of all files from Figshare API"""
    base_url = f"https://api.figshare.com/v2/articles/{figshare_article_id}/files"
    all_files = []
    page = 1
    page_size = 100
    
    try:
        while True:
            params = {'page': page, 'page_size': page_size}
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            files = response.json()
            
            if not files:
                break
            
            all_files.extend(files)
            print(f"  Fetched page {page}: {len(files)} files (total: {len(all_files)})")
            
            if len(files) < page_size:
                break
            
            page += 1
        
        return all_files
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list: {e}")
        sys.exit(1)

def parse_timestamp(timestamp_str):
    """Convert timestamp string to unix timestamp"""
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S"
    ]
    
    timestamp_str = timestamp_str.replace(" UTC", "")
    
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            return dt.timestamp()
        except:
            continue
    
    return 0.0

def extract_page_type_and_title(page_title):
    """Extract page type and clean title from the full page title"""
    namespace_mappings = {
        'User talk:': 'user_talk',
        'User:': 'user',
        'Talk:': 'talk',
        'Wikipedia talk:': 'wikipedia_talk',
        'Wikipedia:': 'wikipedia',
        'File talk:': 'file_talk',
        'File:': 'file',
        'Template talk:': 'template_talk',
        'Template:': 'template',
        'Help talk:': 'help_talk',
        'Help:': 'help',
        'Category talk:': 'category_talk',
        'Category:': 'category',
        'Project talk:': 'project_talk',
        'Project:': 'project',
        'Συζήτηση χρήστη:': 'user_talk',
        'Συζήτηση:': 'talk',
        'Χρήστης:': 'user',
    }
    
    for prefix in sorted(namespace_mappings.keys(), key=len, reverse=True):
        if page_title.startswith(prefix):
            clean_title = page_title[len(prefix):]
            page_type = namespace_mappings[prefix]
            return page_type, clean_title, page_title

    return 'article', page_title, page_title

def process_comment(comment_data):
    """Process a single comment from plain text format to expected format"""
    timestamp = parse_timestamp(comment_data.get('timestamp', ''))
    page_type, clean_title, raw_title = extract_page_type_and_title(
        comment_data.get('page_title', '')
    )
    
    toxicity = float(comment_data.get('toxicity', 0.0))
    sever_toxicity = float(comment_data.get('sever_toxicity', 0.0))
    
    processed_comment = {
        'conversation_id': comment_data.get('conversation_id'),
        'id': comment_data.get('id'),
        'indentation': str(comment_data.get('indentation', 0)),
        'type': comment_data.get('type', 'CREATION').upper(),
        'page_id': str(comment_data.get('page_id', '')),
        'page_title': raw_title,
        'parent_id': comment_data.get('parent_id'),
        'ancestor_id': comment_data.get('ancestor_id'),
        'replyTo_id': comment_data.get('replyTo_id'),
        'rev_id': str(comment_data.get('rev_id', '')),
        'user_id': str(comment_data.get('user_id', '')),
        'user_text': comment_data.get('user_text', ''),
        'toxicity': toxicity,
        'sever_toxicity': sever_toxicity,
        'raw_text': comment_data.get('content', ''),
        'text': comment_data.get('cleaned_content', ''),
        'timestamp': timestamp,
        'is_unchanged': comment_data.get('isUnchanged', False),
        'wiki_links': []
    }
    
    return processed_comment, page_type, clean_title, raw_title

def convert_to_jsonlist(input_path, output_path):
    """Convert a raw file to jsonlist format"""
    conversations = defaultdict(lambda: {
        'comments': [],
        'authors': set(),
        'page_info': {}
    })
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                comment_data = json.loads(line)
                processed_comment, page_type, clean_title, raw_title = process_comment(comment_data)
                
                conv_id = comment_data.get('conversation_id')
                if not conv_id:
                    continue
                
                conversations[conv_id]['comments'].append(processed_comment)
                
                if not conversations[conv_id]['page_info']:
                    conversations[conv_id]['page_info'] = {
                        'page_id': str(comment_data.get('page_id', '')),
                        'page_type': page_type,
                        'page_title': clean_title,
                        'raw_page_title': raw_title
                    }
                
                if 'authors' in comment_data:
                    for author_str in comment_data.get('authors', []):
                        parts = author_str.split(':', 1)
                        if len(parts) == 2:
                            user_id, user_text = parts
                            conversations[conv_id]['authors'].add((user_id, user_text))
                else:
                    user_id = str(comment_data.get('user_id', ''))
                    user_text = comment_data.get('user_text', '')
                    if user_id and user_text:
                        conversations[conv_id]['authors'].add((user_id, user_text))
                        
            except (json.JSONDecodeError, Exception):
                continue
    
    # sort and write convos
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv_id, conv_data in sorted(conversations.items()):
            if not conv_data['comments']:
                continue
            
            creation_comments = [c for c in conv_data['comments'] if c['type'] == 'CREATION']
            other_comments = [c for c in conv_data['comments'] if c['type'] != 'CREATION']
            creation_comments.sort(key=lambda x: (x['timestamp'], x['id']))
            other_comments.sort(key=lambda x: (x['timestamp'], x['id']))
            sorted_comments = creation_comments + other_comments
            
            authors = [
                {'user_text': user_text, 'user_id': user_id}
                for user_id, user_text in sorted(conv_data['authors'])
            ]
            
            conversation = {
                'conversation_id': conv_id,
                'page_id': conv_data['page_info']['page_id'],
                'raw_page_title': conv_data['page_info']['raw_page_title'],
                'page_type': conv_data['page_info']['page_type'],
                'page_title': conv_data['page_info']['page_title'],
                'section_title': None,
                'comments': sorted_comments,
                'authors': authors
            }
            
            json.dump(conversation, f, ensure_ascii=False)
            f.write('\n')
    
    return len(conversations)

def download_and_convert_file(file_info, raw_dir, output_dir, idx, total):
    """Downloads a file, converts it to JSONLIST, and then deletes the raw file."""
    file_name = file_info['name']
    file_url = file_info['download_url']
    raw_path = os.path.join(raw_dir, file_name)
    
    # output filename
    base_name = os.path.splitext(file_name)[0]
    output_filename = f"{base_name}.jsonlist"
    output_path = os.path.join(output_dir, output_filename)
    
    with print_lock:
        print(f"[{idx}/{total}] Processing: {file_name} ({file_info['size'] / (1024*1024):.2f} MB)")
    
    try:
        # download
        response = requests.get(file_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(raw_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with print_lock:
            stats['downloaded'] += 1
        
        conv_count = convert_to_jsonlist(raw_path, output_path)
        
        # delete raw file after successful conversion
        os.remove(raw_path)
        
        with print_lock:
            stats['converted'] += 1
            stats['total_conversations'] += conv_count
            print(f"  ✓ Converted {file_name} → {output_filename} ({conv_count} conversations)")
        
        return (True, conv_count)
        
    except requests.exceptions.RequestException as e:
        with print_lock:
            stats['failed'] += 1
            print(f"  ✗ Download error for {file_name}: {e}")
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return (False, 0)
    except Exception as e:
        with print_lock:
            stats['failed'] += 1
            print(f"  ✗ Processing error for {file_name}: {e}")
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return (False, 0)

def main():
    FIGSHARE_ARTICLE_ID = "7376003"  # English dataset id, change for other datasets
    RAW_DIR = "./raw_data/English"
    OUTPUT_DIR = "./output/English"
    MAX_WORKERS = 10  # adjust as needed
    
    print("=" * 70)
    print("WikiConv Download & Convert Pipeline")
    print("=" * 70)
    print(f"Raw files directory (temporary): {RAW_DIR}")
    print(f"Output JSONLIST directory: {OUTPUT_DIR}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print()
    
    # create directories
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Fetching file list from Figshare...")
    files = get_file_list(FIGSHARE_ARTICLE_ID)
    
    if not files:
        print("No files found!")
        sys.exit(1)
    
    print(f"Found {len(files)} files to process.")
    print()
    
    start_time = time.time()
    
    # process files in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                download_and_convert_file,
                file_info,
                RAW_DIR,
                OUTPUT_DIR,
                idx,
                len(files)
            ): file_info
            for idx, file_info in enumerate(files, 1)
        }
        
        for future in as_completed(future_to_file):
            success, conv_count = future.result()
            completed += 1
            
            # checks every 25 files
            if completed % 25 == 0:
                with print_lock:
                    print(f"\n--- Progress: {completed}/{len(files)} files | "
                          f"Downloaded: {stats['downloaded']} | "
                          f"Converted: {stats['converted']} | "
                          f"Failed: {stats['failed']} | "
                          f"Conversations: {stats['total_conversations']:,} ---\n")
    
    elapsed = time.time() - start_time
    
    try:
        os.rmdir(RAW_DIR)
    except:
        pass
    
    print()
    print("=" * 70)
    print(f"PIPELINE COMPLETED")
    print(f"Files processed: {len(files)}")
    print(f"Successfully downloaded: {stats['downloaded']}")
    print(f"Successfully converted: {stats['converted']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total conversations: {stats['total_conversations']:,}")
    print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Average: {elapsed/len(files):.2f} seconds per file")
    print(f"\nOutput location: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()