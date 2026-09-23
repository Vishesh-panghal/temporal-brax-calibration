#!/bin/bash
# ============================================================================
# Download BRAX PNG images + metadata to external drive
# Then transfer to GPU server via rsync
# ============================================================================
#
# Usage:
#   Step 1 — Download:   bash scripts/download_to_external.sh download
#   Step 2 — Transfer:   bash scripts/download_to_external.sh transfer
#   Check status:        bash scripts/download_to_external.sh status
# ============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
LOCAL_DIR="/Volumes/Vishesh/BRAX-DB"
PHYSIONET_USER="panghal"
PHYSIONET_URL="https://physionet.org/files/brax/1.1.0"

# GPU Server — UPDATE THESE with your actual server details
GPU_HOST="poornima@<GPU_SERVER_IP>"          # e.g. poornima@192.168.1.100
GPU_DEST="/home/poornima/vishesh_gpu/datasets/brax"
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# ─── Step 1: Download to external drive ──────────────────────────────────────
do_download() {
    print_header "BRAX Download → /Volumes/Vishesh/BRAX-DB"

    # Check drive is mounted
    if [ ! -d "$LOCAL_DIR" ]; then
        echo -e "${RED}ERROR: External drive not found at $LOCAL_DIR${NC}"
        echo "Make sure the 'Vishesh' drive is connected and mounted."
        exit 1
    fi

    # Check free space (need ~150GB to be safe)
    FREE_GB=$(df -g "$LOCAL_DIR" | tail -1 | awk '{print $4}')
    echo -e "${GREEN}✓ External drive mounted. Free space: ${FREE_GB} GB${NC}"
    if [ "$FREE_GB" -lt 150 ]; then
        echo -e "${YELLOW}⚠ WARNING: Less than 150 GB free. PNG images need ~115-130 GB.${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 0
    fi

    # Download master CSV first (tiny, 21.7 MB)
    echo -e "\n${YELLOW}[1/3] Downloading master_spreadsheet_update.csv...${NC}"
    wget -N -c --user "$PHYSIONET_USER" --ask-password \
        -P "$LOCAL_DIR" \
        "$PHYSIONET_URL/master_spreadsheet_update.csv"

    # Download SHA256SUMS for integrity verification
    echo -e "\n${YELLOW}[2/3] Downloading SHA256SUMS.txt...${NC}"
    wget -N -c --user "$PHYSIONET_USER" --ask-password \
        -P "$LOCAL_DIR" \
        "$PHYSIONET_URL/SHA256SUMS.txt"

    # Download PNG images (the big one!)
    # -r: recursive  -N: timestamping  -c: continue partial  -np: no parent
    # --cut-dirs=3: strips physionet.org/files/brax/1.1.0/ prefix
    # --reject "index.html*": skip directory listings
    echo -e "\n${YELLOW}[3/3] Downloading PNG images (images/ folder, ~115-130 GB)...${NC}"
    echo -e "${CYAN}This will take several hours depending on your internet speed.${NC}"
    echo -e "${CYAN}Safe to interrupt (Ctrl+C) and resume later — wget will continue from where it left off.${NC}\n"

    wget -r -N -c -np \
        --user "$PHYSIONET_USER" --ask-password \
        --cut-dirs=3 \
        -nH \
        --reject "index.html*" \
        -P "$LOCAL_DIR" \
        "$PHYSIONET_URL/images/"

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ Download complete!${NC}"
    echo -e "${GREEN}  Location: $LOCAL_DIR${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ─── Step 2: Transfer to GPU server ──────────────────────────────────────────
do_transfer() {
    print_header "Transfer BRAX → GPU Server"

    if [[ "$GPU_HOST" == *"<GPU_SERVER_IP>"* ]]; then
        echo -e "${RED}ERROR: Please edit this script and set GPU_HOST to your actual server.${NC}"
        echo "  e.g., GPU_HOST=\"poornima@10.0.0.5\""
        exit 1
    fi

    if [ ! -d "$LOCAL_DIR/images" ]; then
        echo -e "${RED}ERROR: No images/ folder found in $LOCAL_DIR${NC}"
        echo "Run 'bash scripts/download_to_external.sh download' first."
        exit 1
    fi

    # Test SSH connection
    echo -e "${YELLOW}Testing SSH connection to $GPU_HOST...${NC}"
    if ! ssh -o ConnectTimeout=10 "$GPU_HOST" "echo 'SSH OK'" 2>/dev/null; then
        echo -e "${RED}ERROR: Cannot connect to $GPU_HOST via SSH.${NC}"
        echo "Check your SSH key, VPN, or network connection."
        exit 1
    fi
    echo -e "${GREEN}✓ SSH connection OK${NC}\n"

    # Create remote directory
    ssh "$GPU_HOST" "mkdir -p $GPU_DEST"

    # Transfer CSV metadata first
    echo -e "${YELLOW}[1/2] Transferring metadata files...${NC}"
    rsync -avz --progress \
        "$LOCAL_DIR/master_spreadsheet_update.csv" \
        "$GPU_HOST:$GPU_DEST/"

    # Transfer PNG images (the big transfer)
    echo -e "\n${YELLOW}[2/2] Transferring PNG images via rsync (resumable)...${NC}"
    echo -e "${CYAN}This will take a while. Safe to interrupt and resume.${NC}\n"

    rsync -avz --progress \
        --partial --partial-dir=.rsync-partial \
        "$LOCAL_DIR/images/" \
        "$GPU_HOST:$GPU_DEST/images/"

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ Transfer complete!${NC}"
    echo -e "${GREEN}  GPU location: $GPU_HOST:$GPU_DEST${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ─── Check status ────────────────────────────────────────────────────────────
do_status() {
    print_header "BRAX Download Status"

    if [ ! -d "$LOCAL_DIR" ]; then
        echo -e "${RED}External drive not mounted.${NC}"
        exit 1
    fi

    echo -e "${CYAN}Location:${NC} $LOCAL_DIR"
    echo -e "${CYAN}Drive space:${NC}"
    df -h "$LOCAL_DIR" | tail -1

    echo ""
    if [ -d "$LOCAL_DIR/images" ]; then
        IMG_COUNT=$(find "$LOCAL_DIR/images" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        IMG_SIZE=$(du -sh "$LOCAL_DIR/images" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ images/ folder exists${NC}"
        echo -e "  PNG files downloaded: ${CYAN}${IMG_COUNT}${NC}"
        echo -e "  Total size:           ${CYAN}${IMG_SIZE}${NC}"
        echo -e "  Expected:             ~40,967 images"
    else
        echo -e "${YELLOW}✗ images/ folder not found — download not started${NC}"
    fi

    echo ""
    if [ -f "$LOCAL_DIR/master_spreadsheet_update.csv" ]; then
        echo -e "${GREEN}✓ master_spreadsheet_update.csv exists${NC}"
    else
        echo -e "${YELLOW}✗ master_spreadsheet_update.csv not found${NC}"
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    download)
        do_download
        ;;
    transfer)
        do_transfer
        ;;
    status)
        do_status
        ;;
    *)
        echo "Usage: bash scripts/download_to_external.sh {download|transfer|status}"
        echo ""
        echo "  download  — Download BRAX PNG images to external drive"
        echo "  transfer  — rsync downloaded data to GPU server"
        echo "  status    — Check download progress"
        ;;
esac
