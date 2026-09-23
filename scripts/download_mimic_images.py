#!/usr/bin/env python3
"""
Parallel Authenticated Image Downloader for MIMIC-CXR-JPG Stage 4A Test Cohort.
Reads data/processed/selected_IMAGE_FILENAMES.txt and downloads only the required ~5,000
test images from PhysioNet, avoiding the full 567 GB archive.
"""

import os
import sys
import getpass
import argparse
import netrc
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"

# Thread-local storage for requests session
_thread_local = threading.local()


def get_session(username: str, password: str) -> requests.Session:
    """Thread-local session reusing connections."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Wget/1.21.4 (linux-gnu)"})
        s.auth = HTTPBasicAuth(username, password)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=3,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


def resolve_credentials(user=None, password=None):
    """Resolves credentials from arguments, environment variables, ~/.netrc, or prompt."""
    if not user:
        user = os.environ.get("PHYSIONET_USER")
    if not password:
        password = os.environ.get("PHYSIONET_PASSWORD")

    # Check ~/.netrc if still missing
    netrc_path = Path.home() / ".netrc"
    if (not user or not password) and netrc_path.exists():
        try:
            n = netrc.netrc()
            auth = n.authenticators("physionet.org")
            if auth:
                user = user or auth[0]
                password = password or auth[2]
        except Exception:
            pass

    if not user:
        user = "panghal"

    if not password:
        if sys.stdin.isatty():
            print(f"PhysioNet Username: {user}")
            password = getpass.getpass(f"Enter PhysioNet password for '{user}': ")
        else:
            raise ValueError(
                f"Password not provided for PhysioNet user '{user}'. "
                "Provide via --password, set PHYSIONET_PASSWORD environment variable, or configure ~/.netrc."
            )

    return user, password


def is_valid_jpeg(filepath: Path) -> bool:
    """Checks if file exists, is non-empty (>1KB), and has valid JPEG magic bytes."""
    if not filepath.exists() or filepath.stat().st_size < 1024:
        return False
    try:
        with open(filepath, "rb") as f:
            header = f.read(3)
            return header == b"\xff\xd8\xff"
    except Exception:
        return False


def download_single_image(
    rel_path: str,
    output_dir: Path,
    username: str,
    password: str,
    max_retries: int = 3,
) -> tuple:
    """
    Downloads a single JPEG image.
    Returns: (status: str, rel_path: str, error_msg: str or None)
    """
    target_path = output_dir / rel_path

    # Skip if valid image exists
    if is_valid_jpeg(target_path):
        return "SKIPPED", rel_path, None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    url = BASE_URL + rel_path.lstrip("/")

    session = get_session(username, password)

    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=30) as r:
                if r.status_code in (401, 403):
                    return "AUTH_ERROR", rel_path, f"HTTP {r.status_code}: Access Denied"
                if r.status_code == 404:
                    return "NOT_FOUND", rel_path, f"HTTP 404: File not found"
                r.raise_for_status()

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

            # Check file size & magic bytes
            if not is_valid_jpeg(temp_path):
                temp_path.unlink(missing_ok=True)
                if attempt == max_retries - 1:
                    return "CORRUPT", rel_path, "Downloaded file failed JPEG validation"
                continue

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
    parser = argparse.ArgumentParser(description="Parallel download of MIMIC-CXR-JPG Stage 4A test images.")
    parser.add_argument(
        "--file-list",
        type=str,
        default="data/processed/selected_IMAGE_FILENAMES.txt",
        help="Text file listing relative image paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mimic/images",
        help="Destination directory for images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent worker threads (default: 16).",
    )
    parser.add_argument("--user", type=str, default=None, help="PhysioNet username.")
    parser.add_argument("--password", type=str, default=None, help="PhysioNet password.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to download (for testing).")
    args = parser.parse_args()

    file_list_path = Path(args.file_list)
    if not file_list_path.exists():
        print(f"❌ Error: Image list '{file_list_path}' not found. Run scripts/build_mimic_stage4a_manifest.py first.")
        sys.exit(1)

    with open(file_list_path) as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    if args.limit:
        print(f"Limiting to first {args.limit} images for pilot/dry-run.")
        rel_paths = rel_paths[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        user, password = resolve_credentials(args.user, args.password)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n🚀 Preparing parallel download for {len(rel_paths):,} images...")
    print(f"📂 Output directory: {output_dir.resolve()}")
    print(f"⚡ Workers: {args.workers}")

    # Test download single image
    print("\n🔍 Verifying access with pilot test download...")
    status, _, err = download_single_image(rel_paths[0], output_dir, user, password)
    if status == "AUTH_ERROR":
        print(f"❌ Authentication Failed: {err}")
        print("Please check credentials and ensure DUA is signed for MIMIC-CXR-JPG.")
        sys.exit(1)
    print(f"✅ Pilot check succeeded ({status})! Starting parallel pool...\n")

    counts = {"DOWNLOADED": 0, "SKIPPED": 0, "FAILED": 0, "AUTH_ERROR": 0, "NOT_FOUND": 0, "CORRUPT": 0}
    failed_items = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(download_single_image, path, output_dir, user, password): path
            for path in rel_paths
        }

        with tqdm(total=len(rel_paths), desc="Downloading MIMIC-CXR Test Images", unit="img") as pbar:
            for future in as_completed(future_to_path):
                st, path, err = future.result()
                if st in counts:
                    counts[st] += 1
                else:
                    counts["FAILED"] += 1

                if st in ("FAILED", "AUTH_ERROR", "CORRUPT"):
                    failed_items.append((path, err))

                pbar.set_postfix(dl=counts["DOWNLOADED"], skip=counts["SKIPPED"], fail=counts["FAILED"])
                pbar.update(1)

    print("\n" + "=" * 50)
    print("Download Summary:")
    print(f"  Downloaded : {counts['DOWNLOADED']:,}")
    print(f"  Skipped    : {counts['SKIPPED']:,} (already present)")
    print(f"  Failed     : {counts['FAILED'] + counts['CORRUPT']:,}")
    if counts['NOT_FOUND'] > 0:
        print(f"  Not Found  : {counts['NOT_FOUND']:,}")
    print("=" * 50)

    if failed_items:
        print(f"\n⚠️ {len(failed_items)} images failed to download. First 5 errors:")
        for p, err in failed_items[:5]:
            print(f"  {p}: {err}")


if __name__ == "__main__":
    main()
