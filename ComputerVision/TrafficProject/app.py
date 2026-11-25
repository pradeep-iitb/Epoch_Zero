"""
Optional Gradio demo: upload a road photo, get vehicle counts and a congestion level.

    python app.py            # then open the printed local URL (http://127.0.0.1:7860)

Uses the fine-tuned detector in weights/ and the thresholds tuned on the validation set by
traffic_flow_main.ipynb (outputs/final/tuned_config.json). Run that notebook once first.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from trafficlib import (CLASS_NAMES, boxes_in_roi, draw_detections, find_finetuned_weights, load_image_rgb,
                        measure, yolo_detect)

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("TRAFFIC_CONFIG", ROOT / "outputs" / "final" / "tuned_config.json"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not CONFIG.exists():
    raise SystemExit(f"{CONFIG.relative_to(ROOT)} not found. Run traffic_flow_main.ipynb once to tune the thresholds.")
TUNED = json.loads(CONFIG.read_text())
WEIGHTS = ROOT / TUNED["yolo_weights"] if (ROOT / TUNED["yolo_weights"]).exists() else find_finetuned_weights(ROOT)
MODEL = YOLO(str(WEIGHTS))
CFG = {"pcu_weights": TUNED["pcu_weights"], "agnostic_pcu": TUNED["agnostic_pcu"], "alpha": TUNED["alpha"],
       "pcu_sat": TUNED["pcu_sat"], "occ_sat": TUNED["occ_sat"], "thresholds": tuple(TUNED["thresholds"]["yolo"])}

LABELS = {"car": "Cars", "two_wheeler": "Two-wheelers", "auto_rickshaw": "Auto-rickshaws", "bus": "Buses", "truck": "Trucks"}
LEVEL_TEXT = {"free": "Free flow", "moderate": "Moderate", "heavy": "Heavy congestion"}


def signal_html(r: dict) -> str:
    """Traffic-signal meter: the lamp of the measured level is lit and named in text (never colour alone)."""
    lamps = "".join(f'<div class="lamp {lv} {"on" if lv == r["congestion_level"] else ""}" '
                    f'aria-hidden="true"></div>' for lv in ("heavy", "moderate", "free"))
    rows = "".join(f"<tr><th scope='row'>{LABELS[c]}</th><td>{n}</td><td>{r['distribution'][c]:.0%}</td></tr>"
                   for c, n in r["counts"].items())
    return f"""
<div class="meter" role="status" aria-live="polite">
  <div class="signal">{lamps}</div>
  <div class="reading">
    <p class="level">{LEVEL_TEXT[r['congestion_level']]}</p>
    <p class="score">Congestion score <strong>{r['congestion_score']:.2f}</strong> of 1</p>
    <p class="facts"><strong>{r['total']}</strong> vehicles &nbsp; <strong>{r['weighted_count']:.1f}</strong> PCU &nbsp;
       <strong>{r['occupancy']:.0%}</strong> of the road covered</p>
  </div>
</div>
<table class="counts"><thead><tr><th scope="col">Type</th><th scope="col">Count</th><th scope="col">Share</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def analyse(image_path):
    if image_path is None:
        return None, "<p class='hint'>Add a road photo to measure it.</p>", {}
    img = load_image_rgb(image_path)
    h, w = img.shape[:2]
    boxes, scores, cls = yolo_detect(MODEL, img, conf=TUNED["yolo_conf"], imgsz=TUNED["yolo_imgsz"], device=DEVICE)
    result = measure(boxes, cls, w, h, CFG)
    keep = boxes_in_roi(boxes, w, h, None)
    return draw_detections(img, boxes[keep], cls[keep], scores[keep]), signal_html(result), result


CSS = """
:root { --asphalt: #eef1f4; --ink: #1d2329; --sign-blue: #1f5fa8; --go: #0ca30c; --wait: #fab219; --stop: #d03b3b; }
.gradio-container { background: var(--asphalt) !important; max-width: 1180px !important; }
h1.title { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 700; font-size: 2.1rem;
           color: var(--ink); margin: 0.2rem 0 0; letter-spacing: -0.01em; }
p.lede { color: #4a5560; margin: 0.2rem 0 1rem; max-width: 62ch; }
.meter { display: flex; gap: 1.25rem; align-items: center; padding: 1rem; background: #fff;
         border-radius: 14px; border: 1px solid #d5dbe1; }
.signal { display: grid; gap: 8px; padding: 10px; background: var(--ink); border-radius: 12px; }
.lamp { width: 34px; height: 34px; border-radius: 50%; background: #3a434c; transition: background .25s, box-shadow .25s; }
.lamp.on.free { background: var(--go); box-shadow: 0 0 14px var(--go); }
.lamp.on.moderate { background: var(--wait); box-shadow: 0 0 14px var(--wait); }
.lamp.on.heavy { background: var(--stop); box-shadow: 0 0 14px var(--stop); }
.level { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 700; font-size: 1.9rem; margin: 0; color: var(--ink); }
.score, .facts { margin: 0.15rem 0 0; color: #4a5560; }
.score strong, .facts strong { color: var(--ink); font-family: 'Barlow Semi Condensed', sans-serif; font-size: 1.15rem; }
table.counts { width: 100%; margin-top: 0.9rem; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden; }
table.counts th, table.counts td { padding: 0.45rem 0.8rem; border-bottom: 1px solid #e3e7eb; text-align: left; }
table.counts thead th { color: #4a5560; font-weight: 600; font-size: 0.9rem; }
table.counts td { font-variant-numeric: tabular-nums; text-align: right; }
table.counts thead th:not(:first-child) { text-align: right; }
.hint { color: #4a5560; }
button.primary { background: var(--sign-blue) !important; }
button:focus-visible { outline: 3px solid var(--wait) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { .lamp { transition: none; } }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Source Sans 3"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Source Code Pro"), "monospace"],
)
HEAD = '<link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700&display=swap" rel="stylesheet">'

examples = sorted((ROOT / "data" / "test" / "images").glob("*.jpg"))[:3]

with gr.Blocks(title="Road congestion meter") as demo:
    gr.HTML("<h1 class='title'>How congested is this road?</h1>"
            "<p class='lede'>Upload a photo of a road. The model finds every car, two-wheeler, auto-rickshaw, bus "
            "and truck, then rates congestion from the number of vehicles and how much of the road they cover.</p>")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Image(type="filepath", label="Road photo", sources=["upload", "clipboard"])
            btn = gr.Button("Measure traffic", variant="primary")
            out_img = gr.Image(label="Vehicles found", interactive=False)
        with gr.Column(scale=2):
            out_html = gr.HTML("<p class='hint'>Add a road photo to measure it.</p>")
            with gr.Accordion("Raw measurement (JSON)", open=False):
                out_json = gr.JSON()
    if examples:
        gr.Examples([[str(p)] for p in examples], inputs=inp, label="Test images")
    btn.click(analyse, inputs=inp, outputs=[out_img, out_html, out_json])
    inp.upload(analyse, inputs=inp, outputs=[out_img, out_html, out_json])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
