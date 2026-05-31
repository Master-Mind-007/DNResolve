import os
import gzip
from huggingface_hub import HfApi

# This script downloads all chunks from your HF dataset and deduplicates them locally.
# We do this locally because 300 chunks will uncompress to ~150GB of text, 
# which crashes the free 14GB GitHub Actions runners.

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_repo_id:
        print("Set HF_REPO_ID environment variable (e.g., set HF_REPO_ID=username/my-domains)")
        return
        
    api = HfApi(token=hf_token)
    
    print(f"Fetching file list from {hf_repo_id}...")
    files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    chunk_files = [f for f in files if f.startswith("domains_") and f.endswith(".txt.gz")]
    
    print(f"Found {len(chunk_files)} chunks. Starting download and deduplication...")
    
    unique_domains = set()
    master_file = "master_domains_deduplicated.txt"
    
    # We download and process one by one to save disk space
    for i, file_name in enumerate(chunk_files):
        print(f"[{i+1}/{len(chunk_files)}] Downloading {file_name}...")
        local_path = api.hf_hub_download(repo_id=hf_repo_id, filename=file_name, repo_type="dataset", token=hf_token)
        
        print(f"[{i+1}/{len(chunk_files)}] Extracting and deduplicating in memory...")
        with gzip.open(local_path, 'rt', encoding='utf-8') as gz:
            for line in gz:
                d = line.strip()
                if d:
                    unique_domains.add(d)
                    
        # Optional: Delete the local downloaded chunk to save space
        os.remove(local_path)
        
        print(f"-> Unique domains so far: {len(unique_domains):,}")

    print(f"\nWriting {len(unique_domains):,} completely unique domains to {master_file}...")
    with open(master_file, 'w', encoding='utf-8') as f:
        for d in sorted(unique_domains):
            f.write(f"{d}\n")
            
    print("Done! You are now ready to feed this master file into MassDNS.")

if __name__ == '__main__':
    main()
