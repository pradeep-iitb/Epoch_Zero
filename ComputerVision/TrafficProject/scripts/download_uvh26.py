"""
Download a reproducible random subset of the UVH-26 dataset (IISc, CC BY 4.0).

    python scripts/download_uvh26.py --n 3200

What it does
  1. Downloads the COCO-format annotation JSONs (STAPLE consensus labels by default) and the
     repository file list from Hugging Face (dataset iisc-aim/UVH-26, not gated).
  2. Pools the official Train + Val images that have labels, and samples --n of them with a fixed
     seed (42). The official split is ignored on purpose: it is a random per-image split, so the
     same CCTV camera appears in both halves. Our own split (make_splits.py) groups by camera.
  3. Downloads each PNG (~3.4 MB) and stores it as JPEG (quality 90, ~0.4 MB) in
     data/_raw/uvh26/images/<id>.jpg. Already-downloaded images are skipped (resumable).
  4. Writes data/_raw/uvh26/uvh26_subset.json: a COCO file with only the sampled images, whose
     file names point to the JPEGs.

No API key is needed. If you ever need one (rate limits), set the HF_TOKEN environment variable;
it is read from the environment and never written to disk.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO = "https://huggingface.co/datasets/iisc-aim/UVH-26"
API = "https://huggingface.co/api/datasets/iisc-aim/UVH-26"
ANN_FILES = {
    "st": ["UVH-26-Train/UVH-26-ST-Train.json", "UVH-26-Val/UVH-26-ST-Val.json"],
    "mv": ["UVH-26-Train/UVH-26-MV-Train.json", "UVH-26-Val/UVH-26-MV-Val.json"],
}


def http_get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    headers = {}
    if os.environ.get("HF_TOKEN"):  # optional, read from the environment only
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:  # network hiccup -> back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {url.rsplit('/', 1)[-1]}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3200, help="number of images to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labels", choices=["st", "mv"], default="st",
                    help="st = STAPLE consensus (more reliable), mv = majority vote")
    ap.add_argument("--out", default="data/_raw/uvh26")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # 1. Annotations + file list ------------------------------------------------------------
    coco_parts = []
    for rel in ANN_FILES[args.labels]:
        local = out / "annotations" / Path(rel).name
        if not local.exists():
            print("downloading", rel)
            local.write_bytes(http_get(f"{REPO}/resolve/main/{rel}"))
        d = json.loads(local.read_text())
        d["_official_split"] = "train" if "Train" in rel else "val"
        coco_parts.append(d)
    flist = out / "annotations" / "file_list.json"
    if not flist.exists():
        siblings = json.loads(http_get(API))["siblings"]
        flist.write_text(json.dumps([s["rfilename"] for s in siblings if s["rfilename"].endswith(".png")]))
    name_to_repo_path = {Path(p).name: p for p in json.loads(flist.read_text())}

    # 2. Pool and sample ---------------------------------------------------------------------
    categories = coco_parts[0]["categories"]
    images, anns_by_img = [], {}
    for d in coco_parts:
        for a in d["annotations"]:
            anns_by_img.setdefault((d["_official_split"], a["image_id"]), []).append(a)
        for im in d["images"]:
            if im["file_name"] in name_to_repo_path:
                images.append({**im, "official_split": d["_official_split"]})
    images.sort(key=lambda im: im["file_name"])           # deterministic order before sampling
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(images), size=min(args.n, len(images)), replace=False)
    sample = [images[i] for i in sorted(pick)]
    print(f"pool: {len(images)} labelled images -> sampled {len(sample)} (seed {args.seed})")

    # 3. Download + convert to JPEG ------------------------------------------------------------
    def fetch(im):
        dst = out / "images" / (Path(im["file_name"]).stem + ".jpg")
        if dst.exists() and dst.stat().st_size > 0:
            return "skip"
        data = http_get(f"{REPO}/resolve/main/{name_to_repo_path[im['file_name']]}")
        Image.open(io.BytesIO(data)).convert("RGB").save(dst, "JPEG", quality=args.quality)
        return "ok"

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, im): im for im in sample}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                failed.append(futs[f]["file_name"])
                print("FAILED", futs[f]["file_name"], e, file=sys.stderr)
            done += 1
            if done % 100 == 0 or done == len(sample):
                el = time.time() - t0
                print(f"  {done}/{len(sample)} images  ({el / 60:.1f} min, ~{(len(sample) - done) * el / done / 60:.0f} min left)",
                      flush=True)

    # 4. Subset COCO file ----------------------------------------------------------------------
    ok = [im for im in sample if im["file_name"] not in failed]
    new_images = [{**im, "file_name": Path(im["file_name"]).stem + ".jpg"} for im in ok]
    # image ids are unique only within an official split -> re-number them
    remap = {}
    for k, im in enumerate(new_images):
        remap[(im["official_split"], im["id"])] = k + 1
    for im in new_images:
        im["id"] = remap[(im["official_split"], im["id"])]
    final_anns = []
    for im_old in ok:
        for a in anns_by_img.get((im_old["official_split"], im_old["id"]), []):
            final_anns.append({**a, "id": len(final_anns) + 1,
                               "image_id": remap[(im_old["official_split"], im_old["id"])]})
    subset = {"info": {"description": f"UVH-26 random subset (n={len(new_images)}, seed={args.seed}, "
                                      f"labels={args.labels.upper()})",
                       "source": REPO, "license": "CC BY 4.0"},
              "categories": categories, "images": new_images, "annotations": final_anns}
    (out / "uvh26_subset.json").write_text(json.dumps(subset))
    print(f"wrote {out / 'uvh26_subset.json'}: {len(new_images)} images, {len(final_anns)} boxes; "
          f"{len(failed)} failed downloads")


if __name__ == "__main__":
    main()
