#!/usr/bin/env python3
"""Regenerate provenance SHA256 hashes from actual files.

Fixes §4.7: all 11 SHA256 entries in the prior provenance_audit.md 
differed from the current files. This script computes hashes from 
the actual files on disk and writes a verified inventory.
"""

import hashlib
import os
import glob
import datetime

def sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Files to hash: prediction archives, checkpoints, source modules, configs, manifests, publication tables/figures, and manuscript files
    file_patterns = [
        "checkpoints/*.pth",
        "reports/stage2/predictions/preds_*.csv",
        "reports/stage3/stage3_*.csv",
        "reports/manuscript_tables/table*.*",
        "reports/manuscript_figures/fig*.png",
        "manuscript/*.*",
        "docs/stage2/*.md",
        "src/evaluation/mitigation.py",
        "src/evaluation/tdi.py",
        "src/evaluation/metrics.py",
        "src/evaluation/predict_frozen.py",
        "src/evaluation/aggregate_stage2.py",
        "src/evaluation/run_sensitivity_and_cohort_shift.py",
        "src/data/dataset.py",
        "src/data/build_stage2_splits.py",
        "src/training/train_anchor.py",
        "configs/config.yaml",
        "data/processed/stage2_manifest.csv",
        "scripts/run_stage2_matrix.sh",
        "scripts/run_stage3_recalibration.py",
        "scripts/compile_publication_artifacts.py",
        "scripts/regenerate_provenance.py",
    ]
    
    files = []
    for pat in file_patterns:
        full_pat = os.path.join(repo_root, pat)
        matched = glob.glob(full_pat)
        if matched:
            files.extend(sorted(matched))
        else:
            # Single file path
            full_path = os.path.join(repo_root, pat)
            if os.path.isfile(full_path):
                files.append(full_path)
    
    # Compute hashes
    rows = []
    for fp in sorted(set(files)):
        if not os.path.isfile(fp):
            continue
        rel = os.path.relpath(fp, repo_root)
        size = os.path.getsize(fp)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fp)).isoformat()
        sha = sha256_file(fp)
        rows.append((rel, size, mtime, sha))
    
    # Write provenance table
    output_dir = os.path.join(repo_root, "docs/stage2")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "provenance_audit.md")
    
    with open(output_path, "w") as f:
        f.write(f"# Provenance Audit — SHA256 File Inventory\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n")
        f.write(f"**Method:** Machine-generated from `scripts/regenerate_provenance.py`\n\n")
        f.write(f"| File | Size (bytes) | Last Modified | SHA256 |\n")
        f.write(f"|---|---:|---|---|\n")
        for rel, size, mtime, sha in rows:
            f.write(f"| `{rel}` | {size:,} | {mtime} | `{sha[:16]}...{sha[-8:]}` |\n")
        f.write(f"\n**Total files hashed:** {len(rows)}\n")
    
    print(f"✅ Provenance audit written to {output_path} ({len(rows)} files hashed)")
    
    # Also write full hashes to CSV for programmatic comparison
    csv_path = os.path.join(output_dir, "provenance_hashes.csv")
    with open(csv_path, "w") as f:
        f.write("file,size_bytes,last_modified,sha256\n")
        for rel, size, mtime, sha in rows:
            f.write(f"{rel},{size},{mtime},{sha}\n")
    print(f"✅ Full hashes CSV written to {csv_path}")


if __name__ == "__main__":
    main()
