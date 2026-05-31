import argparse
import urllib.request
import gzip
import json
import os
import time
from huggingface_hub import HfApi, CommitOperationAdd

# Maximum runtime before voluntarily shutting down (5.5 hours)
MAX_RUNTIME_SECONDS = 5.5 * 3600 
BATCH_SIZE = 5 # Upload 5 files at a time to avoid HF 128 commits/hour rate limit

def get_latest_crawl_id():
    print("Fetching the latest Common Crawl ID...")
    url = "https://index.commoncrawl.org/collinfo.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        return data[0]['id']

def get_completed_files(api, repo_id):
    """Query HuggingFace to get a list of already uploaded chunks."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return set(files)
    except Exception as e:
        print(f"Warning: Could not fetch files from HF (maybe repo doesn't exist yet?): {e}")
        return set()

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

def upload_batch(api, hf_repo_id, files_to_upload, max_retries=5):
    if not files_to_upload:
        return
        
    print(f"Batch uploading {len(files_to_upload)} files to HuggingFace to save rate limits...")
    operations = []
    for filename in files_to_upload:
        operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=filename))
        
    for attempt in range(max_retries):
        try:
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Batch upload of {len(files_to_upload)} chunks from swarm worker"
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
                sleep_time = 60 # Default for 500s or network drops
                
                if "429" in error_str or "Retry after" in error_str:
                    sleep_time = 300 # Default rate limit sleep
                    import re
                    match = re.search(r"Retry after (\d+) seconds", error_str)
                    if match:
                        sleep_time = int(match.group(1)) + 15 # Wait exactly what they ask + 15s buffer
                    print(f"Rate limited by HuggingFace! Sleeping for {sleep_time} seconds...")
                else:
                    print(f"Network error! Sleeping for {sleep_time} seconds before retrying...")
                    
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

    crawl_id = get_latest_crawl_id()
    print(f"Targeting Crawl: {crawl_id}")
    
    completed_files = get_completed_files(api, hf_repo_id)
    print(f"Found {len(completed_files)} files already on HuggingFace.")

    TOTAL_FILES = 300
    files_to_upload = []
    
    for i in range(TOTAL_FILES):
        # 1. Modulo Sharding
        if i % args.total_workers != args.worker_id:
            continue
            
        output_filename = f"domains_{crawl_id}_part_{i:05d}.txt.gz"
        
        # 2. State Check
        if output_filename in completed_files:
            print(f"[{output_filename}] Already processed by swarm. Skipping.")
            continue
            
        # 3. Time Check
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_RUNTIME_SECONDS:
            print("Reached 5.5 hour runtime limit. Voluntarily shutting down.")
            break
            
        # 4. Extract
        base_url = "https://data.commoncrawl.org/"
        url = base_url + f"cc-index/collections/{crawl_id}/indexes/cdx-{i:05d}.gz"
        
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
            upload_batch(api, hf_repo_id, files_to_upload)
            files_to_upload.clear()

    # Flush any remaining files
    if files_to_upload:
        upload_batch(api, hf_repo_id, files_to_upload)
        files_to_upload.clear()

    print("Worker finished its queue!")

if __name__ == '__main__':
    main()
