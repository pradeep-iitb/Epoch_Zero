# Traffic Density and Congestion Estimation from Single Images

Epoch Zero 2026 · Computer Vision project *"Pixels to Measurements"*.

Given **one photo of a road**, the system detects every vehicle and returns numbers someone can act on:

* vehicles per class (`car`, `two_wheeler`, `auto_rickshaw`, `bus`, `truck`) and in total
* the vehicle-type distribution
* a passenger-car-unit (PCU) weighted count (weights from IRC:106-1990)
* road occupancy: the fraction of the road area covered by vehicles
* a congestion score in [0, 1] and a level: **free / moderate / heavy**

The detector is a COCO-pretrained Ultralytics YOLO (small size; YOLO11s vs YOLO26s chosen on the
validation set) **fine-tuned on Indian traffic images** (UVH-26). It is compared on the same held-out
test set with a **classical OpenCV** baseline and **Grounding DINO used zero-shot**, and all three are
also tested on **Indian road photos from Wikimedia Commons**, a source never used for training.

The full write-up is in [`report/REPORT.pdf`](report/REPORT.pdf) (source: `report/REPORT.md`).

## Results

Held-out UVH-26 test set (300 images, 4,027 vehicles, cameras never used in training):

| | Classical OpenCV | Grounding DINO zero-shot | YOLO26s fine-tuned |
|---|---|---|---|
| mAP@0.5 (5 classes) | n/a | 0.519 | **0.825** |
| Count error (vehicles / image) | 8.25 | 3.21 | **1.81** |
| Congestion level = level from true boxes | 0.257 | 0.733 | **0.817** |

118 Wikimedia Commons photos (10 places, congestion level vs a manual label):

| | Classical OpenCV | Grounding DINO zero-shot | YOLO26s fine-tuned | always "heavy" |
|---|---|---|---|---|
| accuracy | 0.297 | 0.636 | **0.644** | 0.525 |
| macro-F1 | 0.198 | 0.569 | **0.609** | 0.230 |

## Folder layout

```
TrafficProject/
├── traffic_flow_main.ipynb     main notebook: loads weights, evaluates, ends with predict(image_path)
├── trafficlib.py               shared code: measurements, detectors, metrics, drawing
├── app.py                      Gradio demo (python app.py)
├── notebooks/
│   ├── 01_data_prep.ipynb      download, class mapping, leak-free splits, sanity checks, statistics
│   └── 02_train.ipynb          model selection + fine-tuning
├── scripts/                    command-line data steps (see "Run order")
├── data/
│   ├── train/ val/ test/       images/ + labels/ (YOLO format); test = held out, never used for training/tuning
│   ├── own_photos/             Wikimedia Commons photos: raw/ (downloads) · images/ (resized copies) ·
│   │                           commons_sources.csv (author, licence, link) · congestion_labels.csv
│   ├── splits_manifest.csv     every image: source, group (camera), split
│   ├── leakage_report.csv      output of the automated leakage check
│   ├── SOURCES.md              datasets, links, licences, class conversion
│   └── _raw/                   downloaded UVH-26 subset
├── weights/                    fine-tuned model (.pt) + model-selection runs
├── outputs/                    results tables, metrics.json, figures, tuned_config.json, training logs
└── report/                     REPORT.md, REPORT.pdf, figures/
```

## Setup (local, Windows / VS Code)

The project uses the workspace virtual environment `Epoch_Zero/.venv` (Python 3.12.7) and the pinned
packages in the workspace [`requirements.txt`](../../requirements.txt) (CUDA 12.6 build of PyTorch).

```powershell
cd <path>\Epoch_Zero
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Kernel for every notebook: **"Python 3.12 (Epoch_Zero .venv)"**. A GPU is optional for the main
notebook (CPU works, slower); training used an RTX 3050 4 GB laptop GPU.

## Setup (Google Colab)

Upload the `TrafficProject` folder to Google Drive (e.g. `My Drive/TrafficProject`), open
`traffic_flow_main.ipynb` with Colab, choose a GPU runtime and *Run all*. The first cell mounts Drive,
finds the folder (`DRIVE_FOLDER_NAME`) and installs `ultralytics`.

## Run order

| step | what | command / notebook | time (RTX 3050) |
|---|---|---|---|
| 1 | download UVH-26 subset, map classes, split, leakage check | `notebooks/01_data_prep.ipynb` | ~1 h download, then minutes |
| 2 | stage 1: YOLO11s vs YOLO26s on val (8 epochs each, 50 % of train); stage 2: winner on all train | `notebooks/02_train.ipynb` | ~25 min + ~30 min |
| 3 | evaluate the three methods on the test set and on the Commons photos, `predict()` | `traffic_flow_main.ipynb` | minutes |

Scripts behind step 1, in order: `scripts/download_uvh26.py` → `scripts/convert_to_yolo.py` →
`scripts/make_splits.py` (runs `scripts/check_leakage.py`).
External photos: `scripts/collect_commons.py` (download + attribution) and
`scripts/label_congestion.py --split commons` (one congestion label per photo).
`traffic_flow_main.ipynb` never retrains; it loads `weights/*_traffic_best.pt`.
`scripts/make_smoke_data.py` builds a 22-image public sample (`data_smoke/`, coco128) used only by
`SMOKE_TEST = True`, a one-minute code check whose numbers are labelled as not being results.

## Predicting on a new image

Run all cells of `traffic_flow_main.ipynb`, copy the image into the project folder and call
`predict("path/to/image.jpg")`. It draws the detections and returns the counts per class, total,
type distribution, PCU-weighted count, occupancy, congestion score and level. An optional
`roi=[[x, y], ...]` (fractions of width and height) restricts the measurement to the road.
`python app.py` offers the same in a browser.

## Notes

* All constants (PCU weights, score weights, prompts) are in the config cell of
  `traffic_flow_main.ipynb`; thresholds are tuned on **validation** only and saved to
  `outputs/final/tuned_config.json` (used by `predict()` and `app.py`).
* Seeds are fixed to 42. Detector outputs are cached in `outputs/final/cache/` (delete the folder to
  recompute).
* Grounding DINO weights are downloaded from the Hugging Face hub on first use.
* Where GitHub release downloads are blocked, the training notebook falls back to Ultralytics' official
  Hugging Face repositories for the pretrained weights.
* Data licences: UVH-26 is CC BY 4.0; every Commons photo keeps its own licence (CC0 / public domain /
  CC BY / CC BY-SA), listed with its author in `data/own_photos/commons_sources.csv`.
