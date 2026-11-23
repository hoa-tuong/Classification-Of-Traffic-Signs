import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam/resolve/main/traffic-sign-detection-vietnam.zip"
ZIP_PATH = RAW_DIR / "traffic-sign-detection-vietnam.zip"
EXTRACT_DIR = RAW_DIR / "traffic-sign-detection-vietnam"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def main():
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 800 * 1024 * 1024:
        print(f"[+] Downloading real Vietnamese street dataset from Hugging Face ({URL})...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(URL, filename=ZIP_PATH, reporthook=t.update_to)
        print(f"[+] Download complete: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"[+] Found existing complete zip: {ZIP_PATH}")

    print(f"[+] Extracting {ZIP_PATH} to {EXTRACT_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"[+] Extraction complete!")

if __name__ == "__main__":
    main()
