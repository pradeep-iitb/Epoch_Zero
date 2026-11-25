"""
Manually label the congestion level of each image (keyboard tool, resumable).

    python scripts/label_congestion.py --split test     # -> data/test/congestion_labels.csv
    python scripts/label_congestion.py --split val      # optional, lets the notebook tune the
                                                        #   level thresholds on real val labels
    python scripts/label_congestion.py --split commons  # -> data/own_photos/congestion_labels.csv

Keys:  1 = free   2 = moderate   3 = heavy   b = back one image   s = skip   q = save and quit
The CSV (columns: image,level) is saved after every key press, and images that already have a
label are skipped when you restart, so you can label in several sessions.

Labelling guide (judge the ROAD AREA, not the whole picture; do not count boxes):
  free      vehicles move independently, clear gaps of several car lengths, most lanes visibly empty
  moderate  steady stream, gaps of about one car length, some lanes full but traffic is not stuck
  heavy     queues / bumper-to-bumper, the road surface is mostly covered by vehicles
Label from the image alone, BEFORE looking at any model output, so the labels stay independent.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trafficlib import LEVELS, list_images, load_image_rgb  # noqa: E402

KEYS = {ord("1"): "free", ord("2"): "moderate", ord("3"): "heavy"}


def save(path: Path, labels: dict):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "level"])
        for k in sorted(labels):
            w.writerow([k, labels[k]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "commons"],
                    help="'commons' = the Wikimedia Commons photos in data/own_photos/images/<place>/")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--width", type=int, default=1280, help="max display width")
    ap.add_argument("--height", type=int, default=860, help="max display height")
    args = ap.parse_args()

    if args.split == "commons":
        img_dir = Path(args.data) / "own_photos" / "images"
        out = Path(args.data) / "own_photos" / "congestion_labels.csv"
        imgs = sorted(p for p in img_dir.rglob("*.jpg"))
    else:
        img_dir = Path(args.data) / args.split / "images"
        out = Path(args.data) / args.split / "congestion_labels.csv"
        imgs = list_images(img_dir)

    def key(p: Path) -> str:      # commons: "<place>/<file>", split folders: "<file>"
        return p.relative_to(img_dir).as_posix()

    labels = {}
    if out.exists():
        with out.open() as f:
            labels = {r["image"]: r["level"] for r in csv.DictReader(f)}
    todo = [p for p in imgs if key(p) not in labels]
    print(f"{len(imgs)} images, {len(labels)} already labelled, {len(todo)} to go -> {out}")

    i = 0
    while 0 <= i < len(todo):
        p = todo[i]
        img = cv2.cvtColor(load_image_rgb(p), cv2.COLOR_RGB2BGR)
        s = min(args.width / img.shape[1], args.height / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        bar = np.zeros((40, max(img.shape[1], 900), 3), np.uint8)
        cv2.putText(bar, f"[{i + 1}/{len(todo)}]  1=free 2=moderate 3=heavy  b=back s=skip q=quit",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.zeros((img.shape[0], bar.shape[1], 3), np.uint8)
        canvas[:, :img.shape[1]] = img
        cv2.imshow("congestion labelling", np.vstack([bar, canvas]))
        k = cv2.waitKey(0) & 0xFF
        if k in KEYS:
            labels[key(p)] = KEYS[k]
            save(out, labels)
            i += 1
        elif k == ord("b"):
            i = max(0, i - 1)
            labels.pop(key(todo[i]), None)
        elif k == ord("s"):
            i += 1
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    save(out, labels)
    counts = {lv: sum(v == lv for v in labels.values()) for lv in LEVELS}
    print(f"saved {len(labels)} labels to {out}: {counts}")


if __name__ == "__main__":
    main()
