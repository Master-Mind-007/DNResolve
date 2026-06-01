import os
import urllib.request
import gzip
import subprocess
import datetime
from huggingface_hub import HfApi, snapshot_download

def get_expected_part_count(crawl_id):
    """Fetches cc-index.paths.gz to determine exact number of parts."""
    path_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/cc-index.paths.gz"
    req = urllib.request.Request(path_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response, mode='r') as gz:
                content = gz.read().decode('utf-8')
                parts = [p for p in content.strip().split('\n') if p]
                return len(parts)
    except Exception as e:
        print(f"Failed to fetch expected part count for {crawl_id}: {e}")
        return -1

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return
        
    api = HfApi(token=hf_token)
    
    print("Fetching list of all files in the HuggingFace dataset...")
    all_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    
    # Organize files by folder
    crawls = {}
    master_files = []
    
    for file in all_files:
        if "GLOBAL_MASTER" in file:
            continue
        if "/" in file:
            crawl_id, filename = file.split("/", 1)
            if crawl_id.startswith("CC-MAIN-"):
                if crawl_id not in crawls:
                    crawls[crawl_id] = {'chunks': [], 'has_master': False}
                if filename.startswith("master_"):
                    crawls[crawl_id]['has_master'] = True
                    master_files.append(file)
                    crawls[crawl_id]['chunks'].append(file)

    masters_created_this_run = []
    
    # 0. Delete Old Crawls (Dynamic Rolling Window)
    years_to_keep = 3
    current_year = datetime.datetime.now().year
    min_year = current_year - years_to_keep
    
    for crawl_id in list(crawls.keys()):
        try:
            year = int(crawl_id.split('-')[2])
            if year < min_year:
                print(f"[{crawl_id}] Older than {years_to_keep} years. DELETING folder to free space...")
                api.delete_folder(path_in_repo=crawl_id, repo_id=hf_repo_id, repo_type="dataset")
                del crawls[crawl_id] # Remove from active processing
                
                # Also remove its master from master_files if it was there
                master_files = [m for m in master_files if not m.startswith(f"{crawl_id}/")]
        except:
            pass

    # 1. Process Month Masters
    for crawl_id, data in crawls.items():
        if data['has_master']:
            print(f"[{crawl_id}] Master file already exists. Skipping.")
            continue
            
        actual_count = len(data['chunks'])
        expected_count = get_expected_part_count(crawl_id)
        
        print(f"[{crawl_id}] Found {actual_count} / {expected_count} chunks.")
        
        if expected_count > 0 and actual_count == expected_count:
            print(f"[{crawl_id}] Crawl is completely uploaded! Generating month master...")
            
            # Download ONLY this specific folder
            local_dir = f"./temp_{crawl_id}"
            os.makedirs(local_dir, exist_ok=True)
            print(f"Downloading chunks for {crawl_id}...")
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                allow_patterns=f"{crawl_id}/domains_*.txt.gz",
                local_dir=local_dir
            )
            
            # Merge and Deduplicate
            master_filename = f"master_{crawl_id}.txt.gz"
            print(f"Running zcat | sort -u for {crawl_id}...")
            
            # Use shell to run the pipeline
            cmd = f"zcat {local_dir}/{crawl_id}/domains_*.txt.gz | sort -u | gzip > {master_filename}"
            subprocess.run(cmd, shell=True, check=True)
            
            print(f"Uploading {master_filename}...")
            path_in_repo = f"{crawl_id}/{master_filename}"
            api.upload_file(
                path_or_fileobj=master_filename,
                path_in_repo=path_in_repo,
                repo_id=hf_repo_id,
                repo_type="dataset",
                commit_message=f"Upload master deduplicated list for {crawl_id}"
            )
            
            print(f"Cleanup {crawl_id}...")
            os.remove(master_filename)
            subprocess.run(f"rm -rf {local_dir}", shell=True)
            
            masters_created_this_run.append(path_in_repo)
            master_files.append(path_in_repo)
        else:
            print(f"[{crawl_id}] Not complete yet. Waiting for swarm workers to finish.")

    # 2. Process Global Master
    if not masters_created_this_run:
        print("No new month masters were generated. Skipping Global Master generation to save compute.")
        return
        
    print(f"Generating GLOBAL MASTER from {len(master_files)} month masters...")
    
    local_dir = "./temp_global"
    os.makedirs(local_dir, exist_ok=True)
    
    # Download all month masters
    for master_path in master_files:
        print(f"Downloading {master_path}...")
        api.hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=master_path,
            local_dir=local_dir
        )
        
    print(f"Running massive zcat | sort -u and splitting into 50M line chunks to stay under 500MB...")
    
    # 50 million lines uncompressed is ~800MB text, which compresses to ~150MB. 
    # This prevents any individual file from exceeding the 500MB threshold.
    cmd = f"find {local_dir} -name 'master_CC-MAIN-*.txt.gz' -exec zcat {{}} + | sort -u | split -l 50000000 -d - GLOBAL_MASTER_part_"
    subprocess.run(cmd, shell=True, check=True)
    
    print("Compressing all chunks...")
    subprocess.run("gzip GLOBAL_MASTER_part_*", shell=True, check=True)
    
    # Find all generated chunks and upload them
    chunks = [f for f in os.listdir('.') if f.startswith('GLOBAL_MASTER_part_') and f.endswith('.gz')]
    
    for chunk in chunks:
        print(f"Uploading {chunk}...")
        api.upload_file(
            path_or_fileobj=chunk,
            path_in_repo=f"GLOBAL_MASTERS/{chunk}",
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Update GLOBAL MASTER chunk {chunk}"
        )
        os.remove(chunk)
    
    print("Cleanup global temp folder...")
    subprocess.run(f"rm -rf {local_dir}", shell=True)
    
    print("ALL DEDUPLICATION COMPLETE!")

if __name__ == '__main__':
    main()
