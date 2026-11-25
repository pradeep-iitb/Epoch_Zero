"""
Build the train / val / test splits WITHOUT leakage, then run the leakage check.

    python scripts/make_splits.py                     # defaults below
    python scripts/make_splits.py --own-val-locations loc_a,loc_b

Rules
  * Split by SOURCE GROUP, never by individual frame:
      - UVH-26 has ~2,800 CCTV cameras but camera ids were removed from the release, so images are
        grouped into "pseudo-cameras": connected components of the graph that links two images when
        their perceptual hashes (pHash, 64 bit) are within --group-dist bits OR their CNN layout
        descriptors (check_leakage.scene_features) have cosine similarity >= --scene-sim. Both
        thresholds were calibrated by eye in notebooks/01_data_prep.ipynb (pHash <= 12 and CNN >= 0.62
        link frames of the same camera, including day vs night, without chaining unrelated cameras into
        one giant group). Linking is transitive, so frames of one camera end up in one group.
      - Extra photos (data/own_photos/images/<location>/) are grouped by location.
  * UVH-26 groups are shuffled with seed 42 and assigned to train / val / test-pool so the image
    counts approach --fractions (default 0.70 / 0.15 / 0.15).
  * TEST set:
      - --uvh-test-max images of the UVH-26 test pool (300 in this project);
      - extra photos enter the box-level splits only if they have a hand-checked box file in
        data/own_photos/labels/<location>/ (test, unless their location is listed in
        --own-train-locations or --own-val-locations); the Wikimedia Commons photos of this project
        have congestion labels only, so they are not part of the box-level splits;
      - pool images that are not selected are left unused (never moved into train/val), so adding
        test photos later does NOT change train/val and does not require retraining.

Outputs
  data/{train,val,test}/{images,labels}/   images + YOLO labels (file names prefixed with source)
  data/splits_manifest.csv                 file, source, group, split, original path, #boxes
  data/data.yaml                           class names + split folders for Ultralytics
  data/leakage_report.csv                  (from check_leakage.py)

Safety: before copying, the script removes only files it created in a previous run (listed in the
old manifest). Anything else you put in data/<split>/ (e.g. congestion_labels.csv) is left alone.
"""
from __future__ import annotations

import argparse
import collections
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from check_leakage import check, hamming_matrix, phash_int, scene_features  # noqa: E402
from trafficlib import CLASS_NAMES, IMG_EXTS, read_yolo_labels  # noqa: E402

SPLITS = ("train", "val", "test")


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def pseudo_camera_groups(paths: list[Path], group_dist: int, cache: Path | None = None,
                         scene_sim: float | None = None, feat_cache: Path | None = None) -> list[int]:
    """Group images (transitively) that are linked by pHash distance <= group_dist OR, if scene_sim is
    given, by CNN layout similarity >= scene_sim. Returns a group id per image."""
    hashes = {}
    if cache is not None and cache.exists():
        for line in cache.read_text().splitlines():
            name, h = line.split(",")
            hashes[name] = int(h)
    todo = [p for p in paths if p.name not in hashes]
    for k, p in enumerate(todo):
        hashes[p.name] = phash_int(p)
        if (k + 1) % 500 == 0:
            print(f"  hashed {k + 1}/{len(todo)}")
    if cache is not None and todo:
        cache.write_text("\n".join(f"{n},{h}" for n, h in sorted(hashes.items())))
    h = np.array([hashes[p.name] for p in paths], dtype=np.uint64)
    uf = UnionFind(len(paths))
    feats = scene_features(paths, feat_cache) if scene_sim is not None else None
    for s in range(0, len(h), 512):
        link = hamming_matrix(h[s:s + 512], h) <= group_dist
        if feats is not None:
            link |= (feats[s:s + 512] @ feats.T) >= scene_sim
        for r, c in zip(*np.where(link)):
            if s + r < c:
                uf.union(s + r, c)
    roots = [uf.find(i) for i in range(len(paths))]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def assign_groups(group_sizes: dict, fractions: dict, seed: int) -> dict:
    """Shuffle groups (seeded) and greedily give each to the split furthest below its target."""
    rng = np.random.default_rng(seed)
    gids = sorted(group_sizes)
    rng.shuffle(gids)
    total = sum(group_sizes.values())
    target = {s: f * total for s, f in fractions.items()}
    have = {s: 0 for s in fractions}
    out = {}
    for g in gids:
        s = max(fractions, key=lambda k: target[k] - have[k])
        out[g] = s
        have[s] += group_sizes[g]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--uvh-dir", default=None, help="default: <data>/_raw/uvh26")
    ap.add_argument("--fractions", default="0.70,0.15,0.15", help="UVH-26 train,val,test-pool fractions")
    ap.add_argument("--group-dist", type=int, default=12, help="pHash distance that links two UVH images")
    ap.add_argument("--scene-sim", type=float, default=0.62, help="CNN layout similarity that links two UVH images")
    ap.add_argument("--leak-scene-sim", type=float, default=0.70, help="CNN similarity flagged by the leakage check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uvh-test-ratio", type=float, default=0.5)
    ap.add_argument("--uvh-test-max", type=int, default=300)
    ap.add_argument("--own-train-locations", default="")
    ap.add_argument("--own-val-locations", default="")
    ap.add_argument("--leak-dist", type=int, default=8, help="pHash distance used by the leakage check")
    args = ap.parse_args()

    data = Path(args.data)
    uvh = Path(args.uvh_dir) if args.uvh_dir else data / "_raw" / "uvh26"
    fr = dict(zip(("train", "val", "test"), map(float, args.fractions.split(","))))

    # ---------------- UVH-26: pseudo-camera groups -> train / val / test-pool --------------------
    uvh_imgs = sorted(p for p in (uvh / "images").glob("*") if p.suffix.lower() in IMG_EXTS
                      and (uvh / "labels" / (p.stem + ".txt")).exists())
    print(f"UVH-26 images with labels: {len(uvh_imgs)}")
    gids = pseudo_camera_groups(uvh_imgs, args.group_dist, cache=uvh / "phash_cache.csv",
                                scene_sim=args.scene_sim, feat_cache=data / "_cache" / "scene_features.npz")
    sizes = collections.Counter(gids)
    print(f"pseudo-camera groups: {len(sizes)} (largest {max(sizes.values()) if sizes else 0} images, "
          f"{sum(1 for v in sizes.values() if v == 1)} singletons)")
    g_split = assign_groups(dict(sizes), fr, args.seed)

    records = []  # dicts: file, source, group, split, orig_image, orig_label
    pool = []
    for p, g in zip(uvh_imgs, gids):
        rec = {"file": f"uvh26__{p.stem}{p.suffix.lower()}", "source": "uvh26", "group": f"uvh26:cam{g:05d}",
               "orig_image": p, "orig_label": uvh / "labels" / (p.stem + ".txt")}
        if g_split[g] == "test":
            pool.append(rec)
        else:
            records.append({**rec, "split": g_split[g]})

    # ---------------- Own photos: grouped by location ---------------------------------------------
    own_root = data / "own_photos"
    own_train = {s for s in args.own_train_locations.split(",") if s}
    own_val = {s for s in args.own_val_locations.split(",") if s}
    n_own_labelled = n_own_unlabelled = 0
    own_test = 0
    for loc_dir in sorted(p for p in (own_root / "images").glob("*") if p.is_dir()):
        loc = loc_dir.name
        for p in sorted(q for q in loc_dir.glob("*") if q.suffix.lower() in IMG_EXTS):
            lab = own_root / "labels" / loc / (p.stem + ".txt")
            if not lab.exists():
                n_own_unlabelled += 1
                continue
            n_own_labelled += 1
            split = "train" if loc in own_train else "val" if loc in own_val else "test"
            own_test += split == "test"
            records.append({"file": f"own__{loc}__{p.stem}{p.suffix.lower()}", "source": "own",
                            "group": f"own:{loc}", "split": split, "orig_image": p, "orig_label": lab})
    print(f"extra photos: {n_own_labelled} with box labels, {n_own_unlabelled} without (not in the box-level splits)")

    # ---------------- Test pool selection --------------------------------------------------------
    n_uvh_test = (min(len(pool), int(args.uvh_test_ratio * own_test)) if own_test
                  else min(len(pool), args.uvh_test_max))
    rng = np.random.default_rng(args.seed)
    chosen = set(rng.choice(len(pool), size=n_uvh_test, replace=False).tolist()) if n_uvh_test else set()
    for k, rec in enumerate(pool):
        records.append({**rec, "split": "test" if k in chosen else "unused"})
    if not own_test:
        print(f"test set: {n_uvh_test} held-out UVH-26 images")

    # ---------------- Remove files from the previous run (only those we created) ------------------
    manifest = data / "splits_manifest.csv"
    if manifest.exists():
        with manifest.open() as f:
            for r in csv.DictReader(f):
                if r["split"] in SPLITS:
                    for sub, name in (("images", r["file"]), ("labels", Path(r["file"]).stem + ".txt")):
                        (data / r["split"] / sub / name).unlink(missing_ok=True)
    for s in SPLITS:
        for sub in ("images", "labels"):
            (data / s / sub).mkdir(parents=True, exist_ok=True)
            leftovers = [p.name for p in (data / s / sub).iterdir() if p.is_file()]
            if leftovers:
                print(f"WARNING: {len(leftovers)} file(s) not created by this script left in data/{s}/{sub}/")

    # ---------------- Copy + manifest ------------------------------------------------------------
    stats = {s: collections.Counter() for s in SPLITS}
    n_img = collections.Counter()
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "group", "split", "orig_image", "n_boxes"])
        for r in sorted(records, key=lambda r: (r["split"], r["file"])):
            lab = read_yolo_labels(r["orig_label"])
            w.writerow([r["file"], r["source"], r["group"], r["split"],
                        r["orig_image"].relative_to(data).as_posix(), len(lab)])
            if r["split"] not in SPLITS:
                continue
            shutil.copy2(r["orig_image"], data / r["split"] / "images" / r["file"])
            shutil.copy2(r["orig_label"], data / r["split"] / "labels" / (Path(r["file"]).stem + ".txt"))
            n_img[(r["split"], r["source"])] += 1
            for c in lab[:, 0].astype(int):
                stats[r["split"]][CLASS_NAMES[c]] += 1

    (data / "data.yaml").write_text(
        "# Generated by scripts/make_splits.py. Folders are relative to this file.\n"
        "# Notebooks write a runtime copy with an absolute 'path:' for Ultralytics.\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")

    print("\nimages per split and source:")
    for s in SPLITS:
        print(f"  {s:5s}: " + ", ".join(f"{src}={n_img[(s, src)]}" for src in ("uvh26", "own")) +
              f" | boxes: {dict(stats[s])}")
    print(f"  unused UVH-26 test-pool images: {len(pool) - n_uvh_test}")

    print("\nrunning leakage check ...")
    ok = check(data, args.leak_dist, args.leak_scene_sim)["clean"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
