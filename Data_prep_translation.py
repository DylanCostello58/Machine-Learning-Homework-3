"""
Step 1: Download and preprocess the Multi30K dataset.

Multi30K contains ~30,000 English/German sentence pairs split into:
  - train: 29,000 pairs
  - val:    1,014 pairs
  - test:   1,000 pairs

Each language is a separate file where line N in the English file
corresponds to line N in the German file.
"""

import os
import gzip
import urllib.request

# ── Paths (all relative to wherever you run this script from) ─────────────────
DATA_DIR = os.path.join("multi30k")

# Multi30K files hosted on GitHub
BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"

FILES = {
    "train.en": "train.en.gz",
    "train.de": "train.de.gz",
    "val.en":   "val.en.gz",
    "val.de":   "val.de.gz",
    "test.en":  "test_2016_flickr.en.gz",
    "test.de":  "test_2016_flickr.de.gz",
}
# ──────────────────────────────────────────────────────────────────────────────


def download_and_extract(filename: str, save_name: str) -> None:
    """Download a gzipped file and extract it to DATA_DIR."""
    import gzip
    gz_path  = os.path.join(DATA_DIR, filename)
    txt_path = os.path.join(DATA_DIR, save_name)

    if os.path.exists(txt_path):
        print(f"[INFO] Already exists: {save_name}")
        return

    url = BASE_URL + filename
    print(f"[INFO] Downloading {filename}...")
    urllib.request.urlretrieve(url, gz_path)

    print(f"[INFO] Extracting → {save_name}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())

    os.remove(gz_path)  # clean up the .gz file


def load_pairs(src_path: str, tgt_path: str) -> list[tuple[str, str]]:
    """Load parallel sentence pairs from two files."""
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f]
    assert len(src_lines) == len(tgt_lines), "Source and target line counts don't match!"
    return list(zip(src_lines, tgt_lines))


def print_stats(split: str, pairs: list[tuple[str, str]]) -> None:
    src_lens = [len(p[0].split()) for p in pairs]
    tgt_lens = [len(p[1].split()) for p in pairs]
    print(f"\n── {split} split ─────────────────────────────────────")
    print(f"  Pairs          : {len(pairs):,}")
    print(f"  Avg EN length  : {sum(src_lens)/len(src_lens):.1f} words")
    print(f"  Avg DE length  : {sum(tgt_lens)/len(tgt_lens):.1f} words")
    print(f"  Sample EN      : {pairs[0][0]}")
    print(f"  Sample DE      : {pairs[0][1]}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download all splits
    for save_name, filename in FILES.items():
        download_and_extract(filename, save_name)

    # Load and verify each split
    for split in ["train", "val", "test"]:
        src_path = os.path.join(DATA_DIR, f"{split}.en")
        tgt_path = os.path.join(DATA_DIR, f"{split}.de")
        pairs    = load_pairs(src_path, tgt_path)
        print_stats(split, pairs)

    print("\n[DONE] Multi30K downloaded and verified. Next step: vocabulary building.")