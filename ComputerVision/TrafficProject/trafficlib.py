"""
trafficlib.py
=============
Shared code for the project "Traffic Density and Congestion Estimation from Single Images".

It is imported by traffic_flow_main.ipynb, notebooks/01_data_prep.ipynb, notebooks/02_train.ipynb,
the scripts in scripts/ and app.py, so every part of the project computes measurements the same way.
The library holds NO tunable constants: thresholds, weights and paths live in the config cell of
traffic_flow_main.ipynb and are passed in as arguments.

Sections
  1. Class list and label mappings (public dataset labels -> project classes)
  2. Image / label I/O (YOLO format, EXIF-safe loading)
  3. Measurements: per-class counts, type distribution, PCU-weighted count, road occupancy,
     congestion score and congestion level
  4. Detectors: fine-tuned YOLO wrapper, Grounding DINO zero-shot, classical OpenCV baseline
  5. Evaluation: COCO-style mAP, precision/recall, real-unit count errors, threshold tuning
  6. Drawing helpers
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# =============================================================================================
# 1. Classes and label mappings
# =============================================================================================
CLASS_NAMES = ["car", "two_wheeler", "auto_rickshaw", "bus", "truck"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
LEVELS = ["free", "moderate", "heavy"]  # congestion levels, in increasing order

# UVH-26 (IISc, CC BY 4.0) has 14 fine-grained classes. None = dropped.
UVH26_TO_PROJECT = {
    "Hatchback": "car", "Sedan": "car", "SUV": "car", "MUV": "car", "Van": "car",
    "Two-wheeler": "two_wheeler", "Bicycle": "two_wheeler",
    "Three-wheeler": "auto_rickshaw",
    "Bus": "bus", "Mini-bus": "bus", "Tempo-traveller": "bus",
    "Truck": "truck", "LCV": "truck",
    "Other": None, "Others": None,   # the JSON files spell it "Others"
}

# COCO (used for the smoke test and for pre-labelling with a COCO-pretrained model).
# COCO has no auto-rickshaw class, so a COCO model can never predict that class.
COCO_TO_PROJECT = {
    "car": "car", "motorcycle": "two_wheeler", "bicycle": "two_wheeler",
    "bus": "bus", "truck": "truck",
}

SOURCE_MAPPINGS = {"uvh26": UVH26_TO_PROJECT, "coco": COCO_TO_PROJECT}

def _hex(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# Categorical palette (fixed slot order, validated for colour-vision deficiency: adjacent-pair
# CVD dE >= 9.1). Every box also carries a text tag, so colour is never the only cue.
CLASS_HEX = {"car": "#2a78d6", "two_wheeler": "#eb6834", "auto_rickshaw": "#1baf7a",
             "bus": "#eda100", "truck": "#e87ba4", "vehicle": "#6250d6"}  # vehicle = class-agnostic
CLASS_COLORS = {k: _hex(v) for k, v in CLASS_HEX.items()}  # RGB tuples for OpenCV drawing

# One fixed colour per METHOD in every chart (first three slots: valid for all-pairs comparisons).
METHOD_HEX = {"YOLO fine-tuned": "#2a78d6", "Grounding DINO zero-shot": "#eb6834",
              "Classical OpenCV": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_plot_style():
    """Shared matplotlib look: light surface, hairline solid grid, recessive axes, ink-coloured text."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "text.color": INK,
    })


def seq_cmap():
    """Single-hue (blue) light->dark colormap for magnitudes such as confusion matrices."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)


def _text_color_on(rgb) -> tuple:
    """Black or white label text, whichever contrasts more with the fill colour."""
    lum = [c / 255 for c in rgb]
    lum = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in lum]
    L = 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2]
    return (11, 11, 11) if (L + 0.05) / 0.05 > 1.05 / (L + 0.05) else (255, 255, 255)

# =============================================================================================
# 2. Image / label I/O
# =============================================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder) -> list[Path]:
    """All image files directly inside `folder`, sorted by name (deterministic order)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_image_rgb(path) -> np.ndarray:
    """Load an image as an RGB uint8 array, applying the EXIF orientation (phone photos)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


def label_path_for(image_path) -> Path:
    """data/<split>/images/x.jpg -> data/<split>/labels/x.txt (YOLO convention)."""
    p = Path(image_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def read_yolo_labels(path) -> np.ndarray:
    """Read a YOLO label file -> float array (N, 5): class, x_center, y_center, width, height
    (coordinates normalised to [0, 1]). A missing or empty file means 'no vehicles'."""
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array([[float(v) for v in r[:5]] for r in rows], dtype=np.float32)


def yolo_to_xyxy(labels: np.ndarray, w: int, h: int):
    """YOLO rows (N, 5) -> (boxes_xyxy in pixels (N, 4), class ids (N,))."""
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int)
    c, xc, yc, bw, bh = labels.T
    boxes = np.stack([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], 1)
    return boxes.astype(np.float32), c.astype(int)


def xyxy_to_yolo_lines(boxes, cls_ids, w: int, h: int) -> list[str]:
    """Pixel boxes -> YOLO text lines 'cls xc yc bw bh' (normalised, clipped to the image)."""
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes, cls_ids):
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        lines.append(f"{int(c)} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def find_finetuned_weights(project_dir):
    """Path of the fine-tuned detector: the file named in outputs/training/selected_model.json
    (written by notebooks/02_train.ipynb), else the first weights/*_traffic_best.pt.
    SMOKE_ weights are never returned. None if nothing has been trained yet."""
    project_dir = Path(project_dir)
    sel = project_dir / "outputs" / "training" / "selected_model.json"
    if sel.exists():
        p = project_dir / json.loads(sel.read_text())["weights"]
        if p.exists():
            return p
    found = sorted(p for p in (project_dir / "weights").glob("*_traffic_best.pt") if not p.name.startswith("SMOKE"))
    return found[0] if found else None


def load_ground_truth(image_path):
    """Image path -> (boxes_xyxy, class ids, (h, w)) read from the matching YOLO label file."""
    with Image.open(image_path) as im:
        w, h = ImageOps.exif_transpose(im).size
    boxes, cls = yolo_to_xyxy(read_yolo_labels(label_path_for(image_path)), w, h)
    return boxes, cls, (h, w)


# =============================================================================================
# 3. Measurements
# =============================================================================================
def count_per_class(cls_ids, class_names=CLASS_NAMES) -> dict:
    """Number of detections of each class."""
    cls_ids = np.asarray(cls_ids, dtype=int)
    return {name: int((cls_ids == i).sum()) for i, name in enumerate(class_names)}


def type_distribution(counts: dict) -> dict:
    """Fraction of vehicles of each type (all zeros when there are no vehicles)."""
    total = sum(counts.values())
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def pcu_weighted_count(counts: dict, pcu_weights: dict) -> float:
    """Passenger-car-unit weighted vehicle count: sum_c count_c * PCU_c."""
    return float(sum(counts[k] * pcu_weights[k] for k in counts))


def roi_polygon_px(roi, w: int, h: int):
    """ROI given as a list of [x, y] points normalised to [0, 1] -> int pixel polygon.
    roi=None means the full frame."""
    if roi is None:
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
    pts = np.asarray(roi, dtype=np.float32) * np.array([w, h], dtype=np.float32)
    return pts.round().astype(np.int32)


def road_occupancy(boxes_xyxy, w: int, h: int, roi=None, raster_max_side: int = 640) -> float:
    """Fraction of the road ROI covered by the UNION of vehicle boxes (overlaps counted once).
    Computed on a down-scaled raster for speed (exact up to ~1 px at raster scale)."""
    s = min(1.0, raster_max_side / max(w, h))
    ws, hs = max(1, int(round(w * s))), max(1, int(round(h * s)))
    roi_mask = np.zeros((hs, ws), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.round(roi_polygon_px(roi, w, h) * s).astype(np.int32)], 1)
    veh_mask = np.zeros((hs, ws), dtype=np.uint8)
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4) * s:
        xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        xb, yb = min(ws, int(np.ceil(x2))), min(hs, int(np.ceil(y2)))
        if xb > xa and yb > ya:
            veh_mask[ya:yb, xa:xb] = 1
    roi_area = int(roi_mask.sum())
    return float((veh_mask & roi_mask).sum() / roi_area) if roi_area else 0.0


def boxes_in_roi(boxes_xyxy, w: int, h: int, roi=None) -> np.ndarray:
    """Boolean mask: which boxes have their bottom-centre point (the 'ground contact' point)
    inside the ROI. With roi=None every box is kept."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if roi is None or len(boxes_xyxy) == 0:
        return np.ones(len(boxes_xyxy), dtype=bool)
    poly = roi_polygon_px(roi, w, h).astype(np.float32)
    pts = np.stack([(boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2, boxes_xyxy[:, 3]], 1)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts])


def congestion_score(weighted_count: float, occupancy: float, alpha: float,
                     pcu_sat: float, occ_sat: float) -> float:
    """Congestion score in [0, 1]:
        score = alpha * min(1, PCU / PCU_SAT) + (1 - alpha) * min(1, occupancy / OCC_SAT)
    PCU_SAT and OCC_SAT are the values at which each term saturates ("as jammed as it gets")."""
    a = min(1.0, weighted_count / pcu_sat) if pcu_sat > 0 else 0.0
    b = min(1.0, occupancy / occ_sat) if occ_sat > 0 else 0.0
    return float(alpha * a + (1.0 - alpha) * b)


def congestion_level(score: float, thresholds) -> str:
    """score < t1 -> free, t1 <= score < t2 -> moderate, score >= t2 -> heavy."""
    t1, t2 = thresholds
    return LEVELS[0] if score < t1 else (LEVELS[1] if score < t2 else LEVELS[2])


def measure(boxes_xyxy, cls_ids, w: int, h: int, cfg: dict, roi=None, class_agnostic: bool = False,
            thresholds=None) -> dict:
    """All per-image measurements from a set of detections.

    cfg keys used: pcu_weights, agnostic_pcu, alpha, pcu_sat, occ_sat, thresholds.
    Detections whose bottom-centre lies outside the ROI are ignored.
    class_agnostic=True is used for the classical baseline (no class information)."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    cls_ids = np.asarray(cls_ids, dtype=int).reshape(-1)
    keep = boxes_in_roi(boxes_xyxy, w, h, roi)
    boxes_xyxy, cls_ids = boxes_xyxy[keep], cls_ids[keep]
    occ = road_occupancy(boxes_xyxy, w, h, roi)
    if class_agnostic:
        counts = {"vehicle": int(len(boxes_xyxy))}
        distribution = None
        weighted = counts["vehicle"] * float(cfg["agnostic_pcu"])
    else:
        counts = count_per_class(cls_ids)
        distribution = type_distribution(counts)
        weighted = pcu_weighted_count(counts, cfg["pcu_weights"])
    score = congestion_score(weighted, occ, cfg["alpha"], cfg["pcu_sat"], cfg["occ_sat"])
    th = thresholds if thresholds is not None else cfg["thresholds"]
    return {
        "counts": counts,
        "total": int(sum(counts.values())),
        "distribution": distribution,
        "weighted_count": round(weighted, 3),
        "occupancy": round(occ, 4),
        "congestion_score": round(score, 4),
        "congestion_level": congestion_level(score, th),
    }


def load_rois(path) -> dict:
    """Optional ROI overrides file (JSON):
        {"default": null | [[x,y],...],
         "per_location": {"<location>": [[x,y],...]},
         "per_image": {"<image file name>": [[x,y],...]}}
    Points are normalised [0, 1] image coordinates. Missing file -> full frame everywhere."""
    path = Path(path)
    if not path.exists():
        return {"default": None, "per_location": {}, "per_image": {}}
    d = json.loads(path.read_text())
    return {"default": d.get("default"), "per_location": d.get("per_location", {}),
            "per_image": d.get("per_image", {})}


def image_location(image_path) -> str | None:
    """Location id encoded in own-photo file names: own__<location>__<name>.jpg."""
    parts = Path(image_path).stem.split("__")
    return parts[1] if len(parts) >= 3 and parts[0] == "own" else None


def get_roi(image_path, rois: dict):
    """ROI polygon for one image: per-image override > per-location override > default."""
    name = Path(image_path).name
    if name in rois.get("per_image", {}):
        return rois["per_image"][name]
    loc = image_location(image_path)
    if loc is not None and loc in rois.get("per_location", {}):
        return rois["per_location"][loc]
    return rois.get("default")


# =============================================================================================
# 4. Detectors. Each returns (boxes_xyxy [N,4] float, scores [N] float, class ids [N] int).
# =============================================================================================
def _empty_dets():
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), int)


def yolo_detect(model, image_rgb: np.ndarray, conf: float = 0.001, iou: float = 0.7,
                imgsz: int = 960, max_det: int = 300, device=None):
    """Run an Ultralytics YOLO model on one RGB image."""
    # Ultralytics expects OpenCV-style BGR arrays.
    r = model.predict(np.ascontiguousarray(image_rgb[:, :, ::-1]), conf=conf, iou=iou, imgsz=imgsz,
                      max_det=max_det, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return _empty_dets()
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.conf.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(int))


class GroundingDINODetector:
    """Zero-shot detector: Grounding DINO (Hugging Face transformers) prompted with text.

    `prompts` maps each project class to one or more text phrases, e.g.
        {"car": ["car"], "two_wheeler": ["motorcycle", "scooter"], ...}
    Each query box gets, for every phrase, the max sigmoid score over that phrase's tokens.
    The box is assigned the class of its best phrase and scored with that phrase score.
    (This keeps all boxes, unlike the default post-processing that can return empty labels.)"""

    def __init__(self, model_id: str, prompts: dict, device: str = "cpu"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        # Build "phrase1. phrase2. ..." and remember each phrase's class and character span.
        self.phrases, self.phrase_cls, spans, pos = [], [], [], 0
        for cname, plist in prompts.items():
            for ph in ([plist] if isinstance(plist, str) else plist):
                self.phrases.append(ph)
                self.phrase_cls.append(CLASS_TO_ID[cname])
                spans.append((pos, pos + len(ph)))
                pos += len(ph) + 2  # ". "
        self.text = ". ".join(self.phrases) + "."
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        # token index -> phrase index (-1 for [CLS], [SEP], '.')
        self.token_phrase = np.full(len(enc["input_ids"]), -1)
        for t, (a, b) in enumerate(enc["offset_mapping"]):
            if b <= a:
                continue
            for k, (sa, sb) in enumerate(spans):
                if a >= sa and b <= sb:
                    self.token_phrase[t] = k

    def detect(self, image_rgb: np.ndarray, score_threshold: float = 0.05, nms_iou: float = 0.7):
        torch = self.torch
        from torchvision.ops import nms

        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(image_rgb), text=self.text,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        probs = out.logits[0].sigmoid().float().cpu().numpy()          # (queries, text tokens)
        boxes = out.pred_boxes[0].float().cpu().numpy()                 # (queries, 4) cx,cy,w,h in [0,1]
        n_tok = min(probs.shape[1], len(self.token_phrase))
        phrase_scores = np.zeros((probs.shape[0], len(self.phrases)), np.float32)
        for k in range(len(self.phrases)):
            idx = np.where(self.token_phrase[:n_tok] == k)[0]
            if len(idx):
                phrase_scores[:, k] = probs[:, idx].max(1)
        best = phrase_scores.argmax(1)
        scores = phrase_scores[np.arange(len(best)), best]
        keep = scores >= score_threshold
        if not keep.any():
            return _empty_dets()
        cx, cy, bw, bh = boxes[keep].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        xyxy = np.clip(xyxy, 0, [w, h, w, h]).astype(np.float32)
        sc = scores[keep].astype(np.float32)
        cls = np.array(self.phrase_cls)[best[keep]].astype(int)
        # Several queries can lock onto the same vehicle: class-agnostic NMS removes duplicates.
        k = nms(torch.from_numpy(xyxy), torch.from_numpy(sc), nms_iou).numpy()
        return xyxy[k], sc[k], cls[k]


def classical_detect(image_rgb: np.ndarray, params: dict, roi=None):
    """Classical, class-agnostic vehicle 'detector' (no learning):
      grey -> CLAHE contrast boost -> Gaussian blur -> Canny edges -> dilation + closing
      (edges of one vehicle merge into one blob) -> external contours -> filter by area,
      aspect ratio and fill ratio -> bounding boxes -> NMS.
    Works on an image resized to params['work_width'] so the area limits are scale-free.
    Score of a box = fraction of its area covered by the blob (used only to rank boxes for AP).
    params: work_width, canny_low, canny_high, kernel, dilate_iter, min_area, max_area
            (fractions of the image area), min_aspect, max_aspect, min_fill."""
    h, w = image_rgb.shape[:2]
    s = params["work_width"] / w
    img = cv2.resize(image_rgb, (params["work_width"], max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    hs, ws = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (params["kernel"], params["kernel"]))
    blobs = cv2.dilate(edges, k, iterations=params["dilate_iter"])
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, k, iterations=2)
    # Restrict to the road ROI.
    roi_mask = np.zeros((hs, ws), np.uint8)
    cv2.fillPoly(roi_mask, [np.round(roi_polygon_px(roi, w, h) * s).astype(np.int32)], 255)
    blobs = cv2.bitwise_and(blobs, roi_mask)
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(hs * ws)
    boxes, scores = [], []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        a = bw * bh / area_img
        if not (params["min_area"] <= a <= params["max_area"]):
            continue
        ar = bw / max(bh, 1)
        if not (params["min_aspect"] <= ar <= params["max_aspect"]):
            continue
        fill = cv2.contourArea(c) / max(bw * bh, 1)
        if fill < params["min_fill"]:
            continue
        boxes.append([x / s, y / s, (x + bw) / s, (y + bh) / s])
        scores.append(fill)
    if not boxes:
        return _empty_dets()
    boxes, scores = np.array(boxes, np.float32), np.array(scores, np.float32)
    import torch
    from torchvision.ops import nms
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.3).numpy()
    return boxes[keep], scores[keep], np.zeros(len(keep), int)


# =============================================================================================
# 5. Evaluation
# =============================================================================================
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4) in xyxy format."""
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_detections(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, iou_thrs=IOU_THRESHOLDS):
    """COCO-style greedy matching. Predictions are processed from highest to lowest score; each one
    is matched to the unmatched ground-truth box of the same class with the highest IoU.
    Returns tp (N_pred, T) bool, in the ORIGINAL prediction order."""
    n, T = len(pred_boxes), len(iou_thrs)
    tp = np.zeros((n, T), bool)
    if n == 0 or len(gt_boxes) == 0:
        return tp
    ious = box_iou(pred_boxes, gt_boxes)
    same = np.asarray(pred_cls)[:, None] == np.asarray(gt_cls)[None, :]
    ious = np.where(same, ious, 0.0)
    order = np.argsort(-np.asarray(pred_scores), kind="stable")
    for t, thr in enumerate(iou_thrs):
        taken = np.zeros(len(gt_boxes), bool)
        for i in order:
            cand = np.where(~taken, ious[i], 0.0)
            j = int(cand.argmax())
            if cand[j] >= thr:
                tp[i, t] = True
                taken[j] = True
    return tp


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP (same rule as pycocotools): at each recall level
    r = 0, 0.01, ..., 1 take the best precision reached at recall >= r (0 if never reached),
    then average the 101 values. `recall` must be non-decreasing (cumulative)."""
    envelope = np.flip(np.maximum.accumulate(np.flip(precision)))
    levels = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, levels, side="left")
    q = np.zeros(len(levels))
    ok = idx < len(recall)
    q[ok] = envelope[idx[ok]]
    return float(q.mean())


def detection_metrics(preds: list, gts: list, n_classes: int, class_agnostic: bool = False,
                      operating_conf: float = 0.25) -> dict:
    """mAP50, mAP50-95 (COCO 101-point), and precision / recall at the operating confidence
    threshold (IoU 0.5), over a list of images.

    preds: list of (boxes, scores, cls) per image; gts: list of (boxes, cls) per image.
    class_agnostic=True merges every class into a single 'vehicle' class (needed to compare with
    the classical baseline, which has no class information). Classes without any ground-truth box
    are left out of the mean, as in COCO."""
    all_tp, all_conf, all_cls, gt_cls_all = [], [], [], []
    for (pb, ps, pc), (gb, gc) in zip(preds, gts):
        pc = np.zeros(len(pb), int) if class_agnostic else np.asarray(pc, int)
        gc = np.zeros(len(gb), int) if class_agnostic else np.asarray(gc, int)
        all_tp.append(match_detections(pb, ps, pc, gb, gc))
        all_conf.append(np.asarray(ps, np.float32))
        all_cls.append(pc)
        gt_cls_all.append(gc)
    tp = np.concatenate(all_tp) if all_tp else np.zeros((0, 10), bool)
    conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    pcls = np.concatenate(all_cls) if all_cls else np.zeros(0, int)
    gcls = np.concatenate(gt_cls_all) if gt_cls_all else np.zeros(0, int)

    nc = 1 if class_agnostic else n_classes
    ap = np.full((nc, len(IOU_THRESHOLDS)), np.nan)
    for c in range(nc):
        n_gt = int((gcls == c).sum())
        if n_gt == 0:
            continue
        m = pcls == c
        if m.sum() == 0:
            ap[c] = 0.0
            continue
        order = np.argsort(-conf[m], kind="stable")
        tpc = tp[m][order].astype(np.float64)
        ctp, cfp = np.cumsum(tpc, 0), np.cumsum(1 - tpc, 0)
        rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
        for t in range(len(IOU_THRESHOLDS)):
            ap[c, t] = _ap_101(rec[:, t], prec[:, t])
    # Precision / recall at the operating threshold (all classes pooled, IoU 0.5).
    sel = conf >= operating_conf
    n_tp = int(tp[sel, 0].sum())
    precision = n_tp / max(int(sel.sum()), 1)
    recall = n_tp / max(len(gcls), 1)
    valid = ~np.isnan(ap[:, 0])
    return {
        "mAP50": float(np.nanmean(ap[:, 0])) if valid.any() else float("nan"),
        "mAP50_95": float(np.nanmean(ap.mean(1)[valid])) if valid.any() else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "AP50_per_class": {(("vehicle",) if class_agnostic else tuple(CLASS_NAMES))[c]:
                           (float(ap[c, 0]) if valid[c] else None) for c in range(nc)},
        "n_gt": int(len(gcls)),
    }


def count_errors(pred_counts: list, gt_counts: list, class_names=None) -> dict:
    """Real-unit counting errors over images.

    pred_counts / gt_counts: lists of dicts {class: count}. With class_names=None only the total is
    compared (class-agnostic). Returns:
      count_MAE          mean |pred_total - gt_total|          (vehicles per image)
      count_bias         mean (pred_total - gt_total)          (+ = over-counting)
      count_RMSE
      mean_pct_error     mean |pred - gt| / gt * 100 over images with gt > 0
      per_class_MAE      {class: mean |pred_c - gt_c|}
      distribution_L1    mean sum_c |p_c - g_c| over images where both have >= 1 vehicle (0..2)"""
    pt = np.array([sum(p.values()) for p in pred_counts], float)
    gt = np.array([sum(g.values()) for g in gt_counts], float)
    err = pt - gt
    pos = gt > 0
    res = {
        "n_images": int(len(gt)),
        "count_MAE": float(np.abs(err).mean()) if len(gt) else float("nan"),
        "count_bias": float(err.mean()) if len(gt) else float("nan"),
        "count_RMSE": float(np.sqrt((err ** 2).mean())) if len(gt) else float("nan"),
        "mean_pct_error": float((np.abs(err[pos]) / gt[pos]).mean() * 100) if pos.any() else float("nan"),
        "n_images_pct": int(pos.sum()),
        "per_class_MAE": None,
        "distribution_L1": None,
    }
    if class_names is not None:
        res["per_class_MAE"] = {c: float(np.mean([abs(p.get(c, 0) - g.get(c, 0))
                                                  for p, g in zip(pred_counts, gt_counts)]))
                                for c in class_names}
        l1 = []
        for p, g in zip(pred_counts, gt_counts):
            sp, sg = sum(p.values()), sum(g.values())
            if sp > 0 and sg > 0:
                l1.append(sum(abs(p.get(c, 0) / sp - g.get(c, 0) / sg) for c in class_names))
        res["distribution_L1"] = float(np.mean(l1)) if l1 else float("nan")
        res["n_images_L1"] = len(l1)
    return res


def tune_count_threshold(dets: list, gt_totals, grid) -> tuple[float, float]:
    """Pick the confidence threshold that minimises total-count MAE (call on the VALIDATION set).
    dets: list of (boxes, scores, cls). Returns (best_threshold, val_MAE)."""
    gt_totals = np.asarray(gt_totals, float)
    best = (None, np.inf)
    for thr in grid:
        pred = np.array([(np.asarray(s) >= thr).sum() for _, s, _ in dets], float)
        mae = float(np.abs(pred - gt_totals).mean())
        if mae < best[1] - 1e-12:
            best = (float(thr), mae)
    return best


def tune_level_thresholds(scores, labels, grid=None) -> tuple[tuple[float, float], float]:
    """Grid-search (t1, t2), t1 < t2, maximising congestion-level accuracy on the VALIDATION set.
    Ties are broken by macro-F1, then by the wider middle band. Returns ((t1, t2), accuracy)."""
    from sklearn.metrics import f1_score

    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    grid = np.round(np.arange(0.02, 1.0, 0.02), 2) if grid is None else grid
    best_key, best = None, None
    for i, t1 in enumerate(grid):
        for t2 in grid[i + 1:]:
            pred = np.array([congestion_level(s, (t1, t2)) for s in scores])
            acc = float((pred == labels).mean())
            f1 = f1_score(labels, pred, labels=LEVELS, average="macro", zero_division=0)
            key = (acc, f1, t2 - t1)
            if best_key is None or key > best_key:
                best_key, best = key, ((float(t1), float(t2)), acc)
    return best


# =============================================================================================
# 6. Drawing
# =============================================================================================
def draw_detections(image_rgb: np.ndarray, boxes, cls_ids=None, scores=None, agnostic: bool = False,
                    color=None, thickness: int | None = None, show_labels: bool = True,
                    roi=None) -> np.ndarray:
    """Return a copy of the image with boxes drawn (class colour, optional 'name score' tag)."""
    img = image_rgb.copy()
    h, w = img.shape[:2]
    t = thickness or max(1, int(round(max(h, w) / 640)))
    fs = max(0.35, max(h, w) / 2400)
    if roi is not None:
        cv2.polylines(img, [roi_polygon_px(roi, w, h)], True, (255, 255, 0), t)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        name = "vehicle" if (agnostic or cls_ids is None) else CLASS_NAMES[int(cls_ids[i])]
        col = color or CLASS_COLORS[name]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, t)
        if show_labels:
            tag = name if scores is None else f"{name} {float(scores[i]):.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = y1 if y1 - th - 4 >= 0 else y1 + th + 4   # keep the tag inside the image
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty), col, -1)
            cv2.putText(img, tag, (x1 + 1, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, _text_color_on(col), 1,
                        cv2.LINE_AA)
    return img
