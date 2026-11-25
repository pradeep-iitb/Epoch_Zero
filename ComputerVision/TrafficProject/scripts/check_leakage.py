"""
Automated leakage check for the train / val / test splits.

    python scripts/check_leakage.py            # exit code 0 = clean, 1 = leakage found

Checks
  1. Exact duplicates: identical file bytes (MD5) in two different splits.
  2. Near-duplicates: perceptual hash (pHash, 64 bit) Hamming distance <= --max-dist between images
     in two different splits.
  2b. Same scene: cosine similarity >= --scene-sim between CNN layout descriptors (ResNet18 features
     on a 4x7 grid, see scene_features) of images in two different splits. This catches two frames of
     the same camera / two photos of the same spot even when traffic and lighting differ, which pHash
     alone misses (calibrated visually in notebooks/01_data_prep.ipynb).
  3. Group leakage: a source group (pseudo-camera for UVH-26, location for extra photos, recorded in
     data/splits_manifest.csv) appearing in more than one split.

Writes data/leakage_report.csv with every offending pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash_int(path: Path, hash_size: int = 8) -> int:
    """64-bit perceptual hash as a Python int (EXIF orientation is irrelevant for near-dup checks)."""
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("L"), hash_size=hash_size)
    return int(str(h), 16)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 hash vectors."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.bitwise_count(x).astype(np.int16)


def scene_features(paths, cache: Path | None = None, batch: int = 32) -> np.ndarray:
    """'Same place?' descriptor: ImageNet ResNet18 layer-4 feature map of the image resized to
    448x256, average-pooled to a 4x7 grid (keeps the spatial layout of the static scene),
    L2-normalised -> cosine similarity = dot product. Two frames of one fixed camera score high
    even when the traffic and lighting differ. Cached in `cache` (.npz) by file content (MD5), so
    renamed copies (e.g. in data/<split>/images) reuse the same entry."""
    import torch
    import torchvision

    paths = [Path(p) for p in paths]
    keys = [hashlib.md5(p.read_bytes()).hexdigest() for p in paths]
    store = {}
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        store = dict(zip(z["keys"].tolist(), z["feats"]))
    todo = [i for i, k in enumerate(keys) if k not in store]
    if todo:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().to(dev)
        body = torch.nn.Sequential(*list(m.children())[:-2])
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1)
        with torch.no_grad():
            for s in range(0, len(todo), batch):
                idx = todo[s:s + batch]
                x = torch.stack([torch.from_numpy(np.array(Image.open(paths[i]).convert("RGB").resize((448, 256))))
                                 .permute(2, 0, 1).float().to(dev) / 255 for i in idx])
                f = torch.nn.functional.adaptive_avg_pool2d(body((x - mean) / std), (4, 7)).flatten(1)
                f = torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float16)
                for i, v in zip(idx, f):
                    store[keys[i]] = v
                if (s // batch) % 20 == 0:
                    print(f"  scene features {min(s + batch, len(todo))}/{len(todo)}", flush=True)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, keys=np.array(list(store)), feats=np.stack(list(store.values())))
    return np.stack([store[k] for k in keys]).astype(np.float32)


def check(data_dir: Path, max_dist: int = 8, scene_sim: float = 0.70, verbose: bool = True) -> dict:
    files = {s: sorted(p for p in (data_dir / s / "images").glob("*") if p.suffix.lower() in IMG_EXTS)
             for s in SPLITS}
    if verbose:
        print("images per split:", {s: len(v) for s, v in files.items()})
    md5 = {s: [hashlib.md5(p.read_bytes()).hexdigest() for p in files[s]] for s in SPLITS}
    ph = {s: np.array([phash_int(p) for p in files[s]], dtype=np.uint64) for s in SPLITS}
    feats = {s: (scene_features(files[s], data_dir / "_cache" / "scene_features.npz") if files[s] else np.zeros((0, 14336)))
             for s in SPLITS}

    rows = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            # 1. exact duplicates
            ib = {h: k for k, h in enumerate(md5[b])}
            for k, h in enumerate(md5[a]):
                if h in ib:
                    rows.append(["exact_duplicate", a, files[a][k].name, b, files[b][ib[h]].name, 0])
            # 2. near duplicates (chunked to keep memory small)
            if len(ph[a]) and len(ph[b]):
                for s0 in range(0, len(ph[a]), 512):
                    d = hamming_matrix(ph[a][s0:s0 + 512], ph[b])
                    for r, c in zip(*np.where(d <= max_dist)):
                        rows.append(["near_duplicate", a, files[a][s0 + r].name, b, files[b][c].name, int(d[r, c])])
            # 2b. same scene (same camera / place) by CNN layout similarity
            if len(feats[a]) and len(feats[b]):
                sim = feats[a] @ feats[b].T
                for r, c in zip(*np.where(sim >= scene_sim)):
                    rows.append(["same_scene", a, files[a][r].name, b, files[b][c].name, round(float(sim[r, c]), 3)])

    # 3. group leakage from the manifest
    manifest = data_dir / "splits_manifest.csv"
    n_groups_crossing = 0
    if manifest.exists():
        groups = {}
        with manifest.open() as f:
            for r in csv.DictReader(f):
                if r["split"] in SPLITS:
                    groups.setdefault(r["group"], set()).add(r["split"])
        for g, ss in groups.items():
            if len(ss) > 1:
                n_groups_crossing += 1
                rows.append(["group_in_multiple_splits", "|".join(sorted(ss)), g, "", "", ""])

    with (data_dir / "leakage_report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "split_a", "file_a", "split_b", "file_b", "distance_or_similarity"])
        w.writerows(rows)
    summary = {
        "exact_duplicates": sum(r[0] == "exact_duplicate" for r in rows),
        "near_duplicates": sum(r[0] == "near_duplicate" for r in rows),
        "same_scene_pairs": sum(r[0] == "same_scene" for r in rows),
        "groups_crossing_splits": n_groups_crossing,
        "phash_max_dist": max_dist,
        "scene_sim_threshold": scene_sim,
        "n_images": {s: len(v) for s, v in files.items()},
    }
    summary["clean"] = (summary["exact_duplicates"] + summary["near_duplicates"] + summary["same_scene_pairs"]
                        + n_groups_crossing) == 0
    if verbose:
        print("leakage summary:", summary)
        for r in rows[:15]:
            print("  ", r)
        print("PASS: no leakage found" if summary["clean"] else
              f"FAIL: see {data_dir / 'leakage_report.csv'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--max-dist", type=int, default=8, help="pHash Hamming distance for a near-duplicate")
    ap.add_argument("--scene-sim", type=float, default=0.70, help="CNN similarity for 'same scene'")
    args = ap.parse_args()
    sys.exit(0 if check(Path(args.data), args.max_dist, args.scene_sim)["clean"] else 1)


if __name__ == "__main__":
    main()
