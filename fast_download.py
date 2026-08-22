"""Parallel chunk downloader for HuggingFace model.safetensors."""

import concurrent.futures
from pathlib import Path
import sys
import time
import requests

URL = "https://huggingface.co/helen9975/pi05-molmoact-yam/resolve/main/model.safetensors"
OUT_PATH = Path("/home/npow/pi05_checkpoint/model.safetensors")
NUM_WORKERS = 32
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB chunks


def get_file_size(url: str) -> int:
    resp = requests.head(url, allow_redirects=True)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def download_chunk(url: str, start: int, end: int, out_path: Path):
    headers = {"Range": f"bytes={start}-{end}"}
    resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    resp.raise_for_status()
    with open(out_path, "r+b") as f:
        f.seek(start)
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def main():
    print("Getting file size...")
    total_size = get_file_size(URL)
    print(f"Total size: {total_size / (1024**3):.2f} GB ({total_size} bytes)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not OUT_PATH.exists() or OUT_PATH.stat().st_size != total_size:
        with open(OUT_PATH, "wb") as f:
            f.truncate(total_size)

    # Build chunk ranges
    chunks = []
    for start in range(0, total_size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        chunks.append((start, end))

    print(f"Divided into {len(chunks)} chunks of {CHUNK_SIZE // (1024*1024)} MB across {NUM_WORKERS} workers.")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(download_chunk, URL, start, end, OUT_PATH) for start, end in chunks]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            elapsed = time.time() - start_time
            rate = (completed * CHUNK_SIZE) / (1024 * 1024 * elapsed)
            percent = (completed / len(chunks)) * 100
            print(f"\rCompleted {completed}/{len(chunks)} ({percent:.1f}%) - Speed: {rate:.2f} MB/s", end="", flush=True)

    print(f"\nDownload finished in {time.time() - start_time:.2f} s!")


if __name__ == "__main__":
    main()
