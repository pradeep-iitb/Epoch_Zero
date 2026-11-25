# Data sources

Every image used in this project, where it comes from, under which licence, and how its labels were
converted to the project's 5 classes: `car`, `two_wheeler`, `auto_rickshaw`, `bus`, `truck`.

| # | Dataset | Link | Licence | Used for |
|---|---------|------|---------|----------|
| 1 | **UVH-26** (Urban Vision Hackathon dataset, AIM @ IISc, v1.0, 2025) | https://huggingface.co/datasets/iisc-aim/UVH-26 · paper: https://arxiv.org/abs/2511.02563 | **CC BY 4.0** | train, val, and part of test (held-out cameras) |
| 2 | **Wikimedia Commons** photos of Indian roads (collected and screened for this project) | https://commons.wikimedia.org · per-file links in `own_photos/commons_sources.csv` | per file: CC0 / public domain / CC BY / CC BY-SA | external test of the congestion level only (no boxes; not used for training or tuning) |

Candidates considered and not used: UA-DETRAC (Chinese CCTV; no two-wheeler or auto-rickshaw labels),
BDD100K (US dash-cam; no auto-rickshaws), IDD (Indian dash-cam; needs manual registration, research-only
licence, dash-cam viewpoint), Roboflow Universe sets (licences vary per project; usually no
camera/source metadata, so leak-free splitting is hard).

## 1. UVH-26

* **What:** 26,646 frames (1920×1080) from about 2,800 Bengaluru "Safe City" traffic cameras,
  February 2025, 06:00–18:00; faces and number plates masked by the publishers. Boxes for 14 Indian
  vehicle classes, crowd-annotated by 565 students, with two consensus label sets.
* **Citation:** A. Sharma, C. Mhatre, S. Gawali, et al., *Towards Image Annotations and Accurate
  Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0*, Technical Report,
  Indian Institute of Science, arXiv:2511.02563, Nov 2025.
* **What we used:** the **STAPLE** consensus labels (`UVH-26-ST-*.json`, described by the authors as
  the more reliable set; they cover 21,726 of the 26,646 frames). From the pooled official train+val
  frames, **3,200 were sampled at random with seed 42** (`scripts/download_uvh26.py`) and stored as
  JPEG (quality 90). The official train/val split was **not** used because it is a per-frame split
  (the same camera appears on both sides).
* **Class conversion** (`scripts/convert_to_yolo.py`, mapping in `trafficlib.UVH26_TO_PROJECT`):

  | UVH-26 class | → project class |
  |---|---|
  | Hatchback, Sedan, SUV, MUV, Van | car |
  | Two-wheeler, Bicycle | two_wheeler |
  | Three-wheeler | auto_rickshaw |
  | Bus, Mini-bus, Tempo-traveller | bus |
  | Truck, LCV | truck |
  | Others | dropped |

  COCO `[x, y, w, h]` pixel boxes are clipped to the image and written as normalised YOLO
  `class x_center y_center width height`; boxes smaller than 2 px are dropped.
* **Grouping for the split:** camera ids were removed from the public release, so frames are grouped
  into *pseudo-cameras* (perceptual-hash similarity of the static background), and whole groups are
  assigned to train / val / test. See `notebooks/01_data_prep.ipynb`.
* **Attribution (required by CC BY 4.0):** images and annotations © the UVH-26 authors, AIM @ IISc,
  licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes made: subset sampled,
  PNG → JPEG, 14 classes merged into 5.

## 2. Wikimedia Commons photos (external test)

* **Why:** UVH-26 is fixed CCTV in one city. Photos from a different source (hand-held, ground-level,
  several cities) show how the system behaves outside its training domain.
* **Collection** (`scripts/collect_commons.py`, Wikimedia API): the category
  *Traffic congestion in India* and its sub-categories (almost only jams), plus ordinary street and road
  categories for lighter traffic (*Streets in Bengaluru*, *Roads in Bengaluru*, *Streets in Delhi*,
  *Streets in Mumbai*, *Roads in India*). Only freely licensed files at least 800 px wide; downloaded at
  most 1920 px wide into `own_photos/raw/<place>/`, resized copies in `own_photos/images/<place>/`.
* **Screening:** every candidate was inspected; photos without road traffic (portraits, landscapes,
  signs, buildings) and near-identical burst shots were removed.
* **Leakage check:** compared with every train and val image: no near-duplicate (pHash distance ≤ 8)
  and no shared camera view (CNN scene similarity ≥ 0.70).
* **Labels:** one congestion level per photo (free / moderate / heavy), judged by eye from the road area
  before any model output was seen (`scripts/label_congestion.py --split commons` →
  `own_photos/congestion_labels.csv`). No boxes, so these photos measure only the congestion level.
* **Attribution (required by the CC licences):** title, author, licence, licence URL, date and page URL
  of every file in `own_photos/commons_sources.csv`. Changes made: resized to at most 1920 px.
