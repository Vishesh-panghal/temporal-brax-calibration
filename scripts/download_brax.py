#!/usr/bin/env python3
"""
Fast Parallel Downloader for BRAX Chest X-ray Dataset
Directly downloads all PNG images from PhysioNet using credentials,
bypassing slow recursive HTML directory crawling.
"""

import os
import sys
import getpass
import argparse
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

BASE_URL = "https://physionet.org/files/brax/1.1.0/"

# Thread-local storage for requests sessions
_thread_local = threading.local()

def get_session(username, password):
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.auth = HTTPBasicAuth(username, password)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50,
            pool_maxsize=50,
            max_retries=3
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session

def download_file(rel_path, output_dir, username, password, max_retries=3):
    """
    Downloads a single image file.
    Skips if file already exists and is non-empty.
    """
    target_path = output_dir / rel_path
    
    # Skip if already exists and valid size (> 1KB)
    if target_path.exists() and target_path.stat().st_size > 1024:
        return "SKIPPED", rel_path, None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    
    url = BASE_URL + rel_path.lstrip("/")
    session = get_session(username, password)

    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=30) as r:
                if r.status_code == 401 or r.status_code == 403:
                    return "AUTH_ERROR", rel_path, f"HTTP {r.status_code}: Access Denied / Invalid Credentials"
                if r.status_code == 404:
                    return "NOT_FOUND", rel_path, f"HTTP 404: File not found"
                r.raise_for_status()
                
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                
                # Verify file is not empty or an html error
                if temp_path.stat().st_size < 500:
                    with open(temp_path, "r", errors="ignore") as f:
                        content = f.read(200)
                        if "<html" in content.lower():
                            temp_path.unlink(missing_ok=True)
                            return "HTML_ERROR", rel_path, "Received HTML instead of PNG"
                
                # Atomic rename
                temp_path.replace(target_path)
                return "DOWNLOADED", rel_path, None

        except requests.RequestException as e:
            temp_path.unlink(missing_ok=True)
            if attempt == max_retries - 1:
                return "FAILED", rel_path, str(e)
            
    temp_path.unlink(missing_ok=True)
    return "FAILED", rel_path, "Max retries exceeded"

def main():
    parser = argparse.ArgumentParser(description="Download BRAX images in parallel directly from PhysioNet.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/brax_temporal_manifest_updated.csv",
        help="Path to manifest CSV containing PngPath column."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/poornima/vishesh_gpu/datasets/brax",
        help="Destination directory to save images."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent worker threads (default: 16)."
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="PhysioNet username (prompted if omitted)."
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="PhysioNet password (prompted if omitted)."
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest file '{manifest_path}' not found.")
        sys.exit(1)

    print(f"Loading manifest: {manifest_path}...")
    df = pd.read_csv(manifest_path, usecols=["PngPath"])
    image_paths = df["PngPath"].dropna().unique().tolist()
    total_images = len(image_paths)
    print(f"Found {total_images:,} unique images to download.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    username = args.user or input("PhysioNet Username: ").strip()
    password = args.password or getpass.getpass("PhysioNet Password: ")

    if not username or not password:
        print("Error: Username and password are required.")
        sys.exit(1)

    # Test connection with the first image
    print(f"\nVerifying access with test download...")
    test_res, _, err = download_file(image_paths[0], output_dir, username, password)
    if test_res in ("AUTH_ERROR", "HTML_ERROR"):
        print(f"Authentication Failed: {err}")
        print("Please check your PhysioNet credentials and ensure DUA is signed.")
        sys.exit(1)
    print(f"Test download successful ({test_res})! Starting parallel download with {args.workers} workers...\n")

    counts = {"DOWNLOADED": 0, "SKIPPED": 0, "FAILED": 0}
    failed_items = []

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_file, p, output_dir, username, password): p
                for p in image_paths
            }
            
            pbar = tqdm(as_completed(futures), total=total_images, desc="Downloading BRAX", unit="img")
            for future in pbar:
                status, rel_path, err = future.result()
                if status in counts:
                    counts[status] += 1
                else:
                    counts["FAILED"] += 1
                    failed_items.append((rel_path, err))
                
                pbar.set_postfix(
                    done=counts["DOWNLOADED"],
                    skip=counts["SKIPPED"],
                    fail=counts["FAILED"]
                )

    except KeyboardInterrupt:
        print("\nDownload interrupted by user! All progress has been saved. Run again anytime to resume.")
        sys.exit(0)

    print("\nDownload Summary:")
    print(f"  Total Images:      {total_images:,}")
    print(f"  Newly Downloaded:  {counts['DOWNLOADED']:,}")
    print(f"  Already Existed:   {counts['SKIPPED']:,}")
    print(f"  Failed:            {counts['FAILED']:,}")

    if failed_items:
        fail_log = output_dir / "failed_downloads.txt"
        with open(fail_log, "w") as f:
            for p, err in failed_items:
                f.write(f"{p}\t{err}\n")
        print(f"  Failed paths written to: {fail_log}")

if __name__ == "__main__":
    main()
