"""
NYC Taxi MLOps — Utility & Ingestion Helpers
Automated data downloading and resource management.
"""

import os
import urllib.request
from pathlib import Path


def ensure_raw_dataset(filepath: str) -> str:
    """
    Ensure the requested NYC TLC Parquet dataset exists locally.
    If absent, automatically download it from the official NYC TLC CloudFront repository.
    """
    path = Path(filepath)
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    filename = path.name
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[DataIngestion] '{filename}' not found locally. Auto-downloading from NYC TLC: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (MLOps-Continuous-Training-Pipeline)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out_f:
        content = resp.read()
        out_f.write(content)
        size_mb = len(content) / (1024 * 1024)
        print(f"[DataIngestion] Successfully downloaded '{filename}' ({size_mb:.2f} MB) -> {path}\n")

    return str(path)
