#!/usr/bin/env python3
"""
Authenticated Downloader for MIMIC-CXR-JPG v2.1.0 Metadata Files from PhysioNet.
Downloads only the small essential metadata files (~60 MB) to enable storage-efficient
filtering of the official test cohort.
"""

import os
import sys
import getpass
import argparse
import netrc
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"

METADATA_FILES = [
    "SHA256SUMS.txt",
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-split.csv.gz",
    "mimic-cxr-2.0.0-chexpert.csv.gz",
    "mimic-cxr-2.1.0-test-set-labeled.csv",
    "IMAGE_FILENAMES",
]


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

    # Fallback default user if known
    if not user:
        user = "panghal"

    # Interactive prompt if password still missing and running in interactive tty
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


def download_file(filename: str, output_dir: Path, session: requests.Session) -> bool:
    """Downloads a single file from PhysioNet with progress bar and integrity check."""
    target_path = output_dir / filename
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    url = BASE_URL + filename

    print(f"Fetching {filename}...")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            if r.status_code in (401, 403):
                print(f"❌ HTTP {r.status_code}: Access denied for {url}. Check credentials and signed DUA.")
                return False
            if r.status_code == 404:
                print(f"⚠️ HTTP 404: {filename} not found at {url}")
                return False
            r.raise_for_status()

            total_size = int(r.headers.get("content-length", 0))
            with open(temp_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=filename,
                leave=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Check for HTML error payload
        if temp_path.stat().st_size < 1000:
            with open(temp_path, "r", errors="ignore") as f:
                content = f.read(300)
                if "<html" in content.lower():
                    temp_path.unlink(missing_ok=True)
                    print(f"❌ Error: Received HTML error page instead of {filename}.")
                    return False

        temp_path.replace(target_path)
        print(f"✅ Successfully downloaded {filename} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
        return True

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        print(f"❌ Failed to download {filename}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-CXR-JPG v2.1.0 metadata files.")
    parser.add_argument("--output-dir", type=str, default="data/mimic/metadata", help="Destination folder.")
    parser.add_argument("--user", type=str, default=None, help="PhysioNet username (default: 'panghal').")
    parser.add_argument("--password", type=str, default=None, help="PhysioNet password.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        user, password = resolve_credentials(args.user, args.password)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({"User-Agent": "Wget/1.21.4 (linux-gnu)"})
    session.auth = HTTPBasicAuth(user, password)
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    print(f"\n🔐 Authenticating as '{user}' against PhysioNet...")
    print(f"📂 Saving to: {output_dir.resolve()}\n")

    results = {}
    for filename in METADATA_FILES:
        target = output_dir / filename
        if target.exists() and target.stat().st_size > 100:
            print(f"⏩ {filename} already exists ({target.stat().st_size / (1024*1024):.2f} MB). Skipping.")
            results[filename] = True
            continue

        success = download_file(filename, output_dir, session)
        results[filename] = success

    print("\n" + "=" * 50)
    print("Download Summary:")
    for k, v in results.items():
        status = "✅ OK" if v else "❌ FAILED"
        print(f"  {k:40s}: {status}")
    print("=" * 50)

    if not all(results.values()):
        print("\nSome files failed to download. Please check errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All metadata files successfully downloaded!")


if __name__ == "__main__":
    main()
