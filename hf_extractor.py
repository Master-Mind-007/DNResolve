import argparse
import urllib.request
import gzip
import json
import os
import time
from huggingface_hub import HfApi

# Maximum runtime before voluntarily shutting down (5.5 hours)
MAX_RUNTIME_SECONDS = 5.5 * 3600 

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
    
    # Ensure the repo exists (creates a private dataset if it doesn't)
    try:
        api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        print(f"Repo setup note: {e}")

    crawl_id = get_latest_crawl_id()
    print(f"Targeting Crawl: {crawl_id}")
    
    completed_files = get_completed_files(api, hf_repo_id)
    print(f"Found {len(completed_files)} files already on HuggingFace.")

    # A typical crawl has exactly 300 index files: cdx-00000.gz to cdx-00299.gz
    TOTAL_FILES = 300
    
    for i in range(TOTAL_FILES):
        # 1. Modulo Sharding: Only process files assigned to this worker
        if i % args.total_workers != args.worker_id:
            continue
            
        output_filename = f"domains_{crawl_id}_part_{i:05d}.txt.gz"
        
        # 2. State Check: Skip if already uploaded
        if output_filename in completed_files:
            print(f"[{output_filename}] Already processed by swarm. Skipping.")
            continue
            
        # 3. Time Check: Stop if we are close to the 6-hour limit
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_RUNTIME_SECONDS:
            print("Reached 5.5 hour runtime limit. Voluntarily shutting down to avoid GitHub timeout.")
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
                
        # 6. Upload to HuggingFace
        print(f"[{output_filename}] Uploading to HuggingFace {hf_repo_id}...")
        api.upload_file(
            path_or_fileobj=output_filename,
            path_in_repo=output_filename,
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        print(f"[{output_filename}] Upload complete!")
        
        # 7. Cleanup local file to save runner disk space
        os.remove(output_filename)

    print("Worker finished its queue!")

if __name__ == '__main__':
    main()
