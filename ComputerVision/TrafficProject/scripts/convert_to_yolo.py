"""
Convert public-dataset labels to the project's 5 classes in YOLO format.

Project classes (id: name): 0 car, 1 two_wheeler, 2 auto_rickshaw, 3 bus, 4 truck.
Mappings live in trafficlib.SOURCE_MAPPINGS (merge several source classes into one project class,
or drop a class with None). Current mappings:

  uvh26: Hatchback/Sedan/SUV/MUV/Van -> car, Two-wheeler/Bicycle -> two_wheeler,
         Three-wheeler -> auto_rickshaw, Bus/Mini-bus/Tempo-traveller -> bus,
         Truck/LCV -> truck, Others -> dropped
  coco:  car -> car, motorcycle/bicycle -> two_wheeler, bus -> bus, truck -> truck,
         every other COCO class -> dropped

Two input formats:
  --format coco   a COCO JSON file (bbox = [x, y, w, h] in pixels)
  --format yolo   an existing YOLO label folder with its own class list (--names file or a
                  comma-separated list), e.g. COCO-80 YOLO labels from Ultralytics' coco128

Examples
  python scripts/convert_to_yolo.py --source uvh26 --format coco \
      --coco data/_raw/uvh26/uvh26_subset.json --out-labels data/_raw/uvh26/labels
  python scripts/convert_to_yolo.py --source coco --format yolo --names coco80 \
      --in-labels <coco128>/labels/train2017 --out-labels data_smoke/_raw/labels

Images with no remaining boxes still get an empty label file (= "no vehicles", a valid negative).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make trafficlib importable
from trafficlib import CLASS_NAMES, CLASS_TO_ID, SOURCE_MAPPINGS  # noqa: E402

# The 80 COCO class names in Ultralytics' YOLO order (used by coco8 / coco128).
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
          "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]


def convert_coco(coco_json: Path, mapping: dict, out_dir: Path, min_box_px: float = 2.0):
    d = json.loads(coco_json.read_text())
    cat_name = {c["id"]: c["name"] for c in d["categories"]}
    unknown = sorted({n for n in cat_name.values() if n not in mapping})
    if unknown:
        raise SystemExit(f"Classes without a mapping (add them to trafficlib.SOURCE_MAPPINGS): {unknown}")
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    stats_src, stats_dst, n_small = collections.Counter(), collections.Counter(), 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for im in d["images"]:
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img.get(im["id"], []):
            if a.get("iscrowd", 0):
                continue
            src = cat_name[a["category_id"]]
            stats_src[src] += 1
            dst = mapping[src]
            if dst is None:
                continue
            x, y, w, h = a["bbox"]
            # clip to the image and drop degenerate boxes
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(W), x + w), min(float(H), y + h)
            if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
                n_small += 1
                continue
            stats_dst[dst] += 1
            lines.append(f"{CLASS_TO_ID[dst]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (out_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(d["images"]), stats_src, stats_dst, n_small


def convert_yolo(in_dir: Path, names: list, mapping: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_src, stats_dst, n = collections.Counter(), collections.Counter(), 0
    for f in sorted(in_dir.glob("*.txt")):
        n += 1
        lines = []
        for row in f.read_text().split("\n"):
            parts = row.split()
            if len(parts) < 5:
                continue
            src = names[int(parts[0])]
            stats_src[src] += 1
            dst = mapping.get(src)  # classes not listed are dropped
            if dst is None:
                continue
            stats_dst[dst] += 1
            lines.append(" ".join([str(CLASS_TO_ID[dst])] + parts[1:5]))
        (out_dir / f.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    return n, stats_src, stats_dst, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_MAPPINGS))
    ap.add_argument("--format", required=True, choices=["coco", "yolo"])
    ap.add_argument("--coco", help="COCO JSON (for --format coco)")
    ap.add_argument("--in-labels", help="YOLO label folder (for --format yolo)")
    ap.add_argument("--names", default="coco80",
                    help="'coco80', a text file with one class name per line, or a comma-separated list")
    ap.add_argument("--out-labels", required=True)
    args = ap.parse_args()

    mapping = SOURCE_MAPPINGS[args.source]
    out = Path(args.out_labels)
    if args.format == "coco":
        n, s_src, s_dst, n_small = convert_coco(Path(args.coco), mapping, out)
    else:
        if args.names == "coco80":
            names = COCO80
        elif Path(args.names).exists():
            names = [l.strip() for l in Path(args.names).read_text().splitlines() if l.strip()]
        else:
            names = [s.strip() for s in args.names.split(",")]
        n, s_src, s_dst, n_small = convert_yolo(Path(args.in_labels), names, mapping, out)

    print(f"converted {n} label files -> {out}")
    print("source class counts :", dict(s_src.most_common()))
    print("project class counts:", {c: s_dst.get(c, 0) for c in CLASS_NAMES})
    dropped = sum(s_src.values()) - sum(s_dst.values()) - n_small
    print(f"dropped (unmapped classes): {dropped} boxes | dropped (degenerate < 2 px): {n_small} boxes")


if __name__ == "__main__":
    main()
