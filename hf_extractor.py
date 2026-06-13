import argparse
import urllib.request
import gzip
import json
import os
import time
import datetime
import random
from huggingface_hub import HfApi, CommitOperationAdd

# Maximum runtime before voluntarily shutting down (5.5 hours)
MAX_RUNTIME_SECONDS = 5.5 * 3600 
BATCH_SIZE = 15 # Upload 15 files at a time to slash total HF API requests
CRAWL_INDEX_LIMIT = 2

def get_all_crawls(years_to_keep=3):
    current_year = datetime.datetime.now().year
    min_year = current_year - years_to_keep
    print(f"Fetching the list of all historical Common Crawl IDs (since {min_year})...")
    url = "https://index.commoncrawl.org/collinfo.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        # Returns list of dicts, newest first
        crawls = []
        for c in data:
            try:
                year = int(c['id'].split('-')[2])
                if year >= min_year:
                    crawls.append(c['id'])
            except:
                pass
        return crawls

def get_completed_files(api, repo_id):
    """Query HuggingFace to get a list of already uploaded chunks."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return set(files)
    except Exception as e:
        print(f"Warning: Could not fetch files from HF (maybe repo doesn't exist yet?): {e}")
        return set()

def get_crawl_paths(crawl_id):
    """Fetches cc-index.paths.gz to determine exact number of parts."""
    path_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/cc-index.paths.gz"
    req = urllib.request.Request(path_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response, mode='r') as gz:
                content = gz.read().decode('utf-8')
                parts = [p for p in content.strip().split('\n') if 'cdx-' in p and p.endswith('.gz')]
                return parts
    except Exception as e:
        print(f"Failed to fetch paths for {crawl_id}: {e}")
        return []

def stream_and_extract(url):
    """Streams a single CDX file and returns a set of unique domains."""
    print(f"Streaming {url}...")
    unique_domains = set()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response, mode='r') as gz:
                for line_bytes in gz:
                    try:
                        line = line_bytes.decode('utf-8', errors='ignore')
                        surt = line.split(' ')[0]
                        if ')/' in surt:
                            host_part = surt.split(')/')[0]
                            parts = host_part.split(',')
                            domain = '.'.join(reversed(parts))
                            unique_domains.add(domain)
                    except Exception:
                        continue
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None
        
    return unique_domains

def upload_batch(api, hf_repo_id, files_to_upload, crawl_id, max_retries=5):
    if not files_to_upload:
        return
        
    print(f"Batch uploading {len(files_to_upload)} files to HuggingFace ({crawl_id})...")
    operations = []
    for filename in files_to_upload:
        # Upload into the crawl's dedicated folder
        path_in_repo = f"{crawl_id}/{filename}"
        operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=filename))
        
    for attempt in range(max_retries):
        try:
            # ANTI-COLLISION JITTER: Stagger the 20 workers randomly so they don't hit HF API at the exact same moment
            jitter = random.uniform(5, 45)
            print(f"Applying {jitter:.1f}s anti-collision jitter before commit...")
            time.sleep(jitter)
            
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Batch upload of {len(files_to_upload)} chunks for {crawl_id}"
            )
            print("Batch upload complete!")
            # Cleanup
            for filename in files_to_upload:
                try:
                    os.remove(filename)
                except OSError:
                    pass
            return # Success!
            
        except Exception as e:
            error_str = str(e)
            print(f"Upload failed (Attempt {attempt+1}/{max_retries}): {error_str}")
            if attempt < max_retries - 1:
                # EXPONENTIAL BACKOFF
                base_sleep = 60
                
                if "429" in error_str or "Retry after" in error_str:
                    base_sleep = 300
                    import re
                    match = re.search(r"Retry after (\d+) seconds", error_str)
                    if match:
                        base_sleep = int(match.group(1)) + 15 
                    
                    # Apply exponential multiplier based on attempt number (e.g. 300s -> 600s -> 1200s)
                    sleep_time = base_sleep * (2 ** attempt)
                    print(f"Rate limited by HuggingFace! Exponential backoff sleeping for {sleep_time} seconds...")
                else:
                    sleep_time = base_sleep * (2 ** attempt)
                    print(f"Network error! Exponential backoff sleeping for {sleep_time} seconds before retrying...")
                    
                time.sleep(sleep_time)
            else:
                print(f"CRITICAL: Failed to upload batch after {max_retries} attempts.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True, help="ID of this worker (e.g., 0 to 19)")
    parser.add_argument('--total-workers', type=int, required=True, help="Total number of workers (e.g., 20)")
    args = parser.parse_args()

    start_time = time.time()
    
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return
        
    api = HfApi(token=hf_token)
    
    # Ensure the repo exists
    try:
        api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        pass

    all_crawls = get_all_crawls()
    hf_files = get_completed_files(api, hf_repo_id)
    
    # Identify active crawls
    # A crawl is considered "completed" if its master file exists in its folder.
    active_crawls = []
    for crawl_id in all_crawls:
        master_file_path = f"{crawl_id}/master_{crawl_id}.txt.gz"
        if master_file_path not in hf_files:
            active_crawls.append(crawl_id)
            if len(active_crawls) >= CRAWL_INDEX_LIMIT: # Limit to 3 active crawls per swarm run to not spread too thin
                break
                
    print(f"Targeting {len(active_crawls)} active crawls: {active_crawls}")

    for crawl_id in active_crawls:
        print(f"--- Starting work on {crawl_id} ---")
        paths = get_crawl_paths(crawl_id)
        if not paths:
            continue
            
        total_parts = len(paths)
        print(f"Crawl {crawl_id} has {total_parts} exact parts.")
        
        files_to_upload = []
        
        for i in range(total_parts):
            # 1. Modulo Sharding
            if i % args.total_workers != args.worker_id:
                continue
                
            output_filename = f"domains_{crawl_id}_part_{i:05d}.txt.gz"
            path_in_repo = f"{crawl_id}/{output_filename}"
            
            # 2. State Check
            if path_in_repo in hf_files:
                print(f"[{output_filename}] Already on HuggingFace. Skipping.")
                continue
                
            # 3. Time Check
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_RUNTIME_SECONDS:
                print("Reached 5.5 hour runtime limit. Voluntarily shutting down.")
                break
                
            # 4. Extract
            base_url = "https://data.commoncrawl.org/"
            url = base_url + paths[i]
            
            domains = stream_and_extract(url)
            if domains is None:
                continue
                
            print(f"[{output_filename}] Extracted {len(domains):,} unique domains. Compressing...")
            
            # 5. Compress locally
            with gzip.open(output_filename, 'wt', encoding='utf-8') as f:
                for d in sorted(domains):
                    f.write(f"{d}\n")
                    
            files_to_upload.append(output_filename)
            
            # 6. Batch Upload Logic
            if len(files_to_upload) >= BATCH_SIZE:
                upload_batch(api, hf_repo_id, files_to_upload, crawl_id)
                files_to_upload.clear()

        # Flush any remaining files for this crawl
        if files_to_upload:
            upload_batch(api, hf_repo_id, files_to_upload, crawl_id)
            files_to_upload.clear()
            
        # Time check between crawls
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            break

    print("Worker finished its queue!")

if __name__ == '__main__':
    main()
