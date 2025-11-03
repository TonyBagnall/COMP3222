#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, requests
from urllib.parse import urlsplit
from ucimlrepo import fetch_ucirepo

OUT_DIR = r"C:\Temp\UCI_All" if len(sys.argv) < 2 else sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "UCI-Zip-Grabber/1.0 (+python requests)"}

def safe_name(name: str) -> str:
    name = name or ""
    return "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name).strip() or "dataset"

def last_segment(url: str) -> str:
    path = urlsplit(url or "").path.rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""

def candidate_zip_urls(meta):
    """
    Build a small set of plausible ZIP URLs for a dataset.
    Primary pattern (most common):
      https://archive.ics.uci.edu/static/public/{id}/{slug}.zip
    Also try a few slug normalisations and a '/download' endpoint.
    """
    uci_id = getattr(meta, "uci_id", None)
    repo = getattr(meta, "repository_url", None) or ""
    slug = last_segment(repo)
    candidates = []
    norm_slugs = {
        slug,
        slug.lower(),
        slug.replace(" ", "-"),
        slug.replace(" ", "_"),
        slug.lower().replace(" ", "-"),
        slug.lower().replace(" ", "_"),
        slug.replace("-", "_"),
        slug.replace("_", "-"),
    }
    for s in list(norm_slugs):
        candidates.append(f"https://archive.ics.uci.edu/static/public/{uci_id}/{s}.zip")
    # Fallback: some pages support a direct download endpoint
    if slug:
        candidates.append(f"https://archive.ics.uci.edu/dataset/{uci_id}/{slug}/download")
        candidates.append(f"https://archive.ics.uci.edu/dataset/{uci_id}/{slug}?download=1")
    return list(dict.fromkeys(candidates))  # de-dup preserving order

def looks_like_zip(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    cd = resp.headers.get("Content-Disposition") or ""
    # Common server types for zips; some serve as octet-stream
    if "zip" in ct:
        return True
    if "application/octet-stream" in ct and ("filename=" in cd or resp.content[:2] == b"PK"):
        return True
    # Heuristic: ZIP files start with "PK"
    return resp.content[:2] == b"PK"

def try_fetch_zip(url: str, timeout=60):
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code == 200 and looks_like_zip(r):
            return r
        return None
    except requests.RequestException:
        return None

def main():
    seen, downloaded = 0, 0
    # Iterate over a superset of current IDs; many will be empty
    for ds_id in range(1, 900):
        try:
            ds = fetch_ucirepo(id=ds_id)  # returns data + metadata + variables
        except Exception:
            continue  # unused ID
        meta = ds.metadata
        seen += 1

        name = safe_name(getattr(meta, "name", None)) or f"dataset_{ds_id}"
        out_path = os.path.join(OUT_DIR, f"{name}.zip")

        # Skip if already present (idempotent)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[skip-existing] {name}")
            continue

        urls = candidate_zip_urls(meta)
        print(f"→ {name}: trying {len(urls)} URL(s)")
        saved = False
        for u in urls:
            r = try_fetch_zip(u)
            if r is None:
                continue
            # Prefer server-provided filename if present
            filename = None
            cd = r.headers.get("Content-Disposition") or ""
            if "filename=" in cd:
                filename = cd.split("filename=", 1)[1].strip(' ";')
            path = os.path.join(OUT_DIR, filename) if filename else out_path
            with open(path, "wb") as f:
                f.write(r.content)
            downloaded += 1
            print(f"[{downloaded}] saved ZIP from {u} -> {path}")
            saved = True
            break

        if not saved:
            # Tell us what failed (first two URLs for brevity)
            preview = " | ".join(urls[:2])
            print(f"[no-zip] {name}: none of the candidate URLs worked. e.g. {preview}")
        time.sleep(0.15)  # be gentle with the server

    print(f"\nScanned {seen} valid dataset IDs; downloaded {downloaded} ZIPs to:\n  {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
