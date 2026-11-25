# Epoch_Zero

Coursework and capstone projects in computer vision and natural language processing.

| Folder | Contents |
|---|---|
| [ComputerVision/TrafficProject](ComputerVision/TrafficProject) | CV capstone "Pixels to Measurements": vehicle detection and congestion estimation on Indian roads (YOLO26s fine-tuned on UVH-26, compared with Grounding DINO and a classical baseline) |
| [ComputerVision](ComputerVision) | CV assignments 1–3 (notebooks and scripts) |
| [NaturalLanguageProcessing/SignalDesk](https://github.com/pradeep-iitb/SignalDesk) | NLP capstone "The Signal Desk": arXiv area classifier with a time-shift stress test (submodule) |
| [NaturalLanguageProcessing](NaturalLanguageProcessing) | NLP assignments 1–2 |
| `TrafficProject_Explained.pdf`, `SignalDesk_Explained.pdf` | Study guides for the two capstones |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
python -m pip install -r requirements.txt
git submodule update --init   # fetch SignalDesk
```

Large data (UVH-26, the Commons photos and the course datasets) is not stored in git. Each project's README explains how to download it again.
