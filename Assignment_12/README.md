---
title: Pakistani Traffic Object Detection & Tracking
emoji: 🚦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
short_description: Enterprise Traffic AI for Pakistani roads with real-time tracking
---

# 🚦 Assignment 12 — Pakistani Traffic Object Detection & Multi-Object Tracking

A Computer Vision application built with **Ultralytics YOLO** and **Gradio**, trained for Pakistani traffic (rickshaws, bikes, buses, cars and pedestrians).

> Based on [murtazakhanpro/Assignment_SMIT_AIDS_2026 — Assignment_12_ObjectDetection_CV](https://github.com/murtazakhanpro/Assignment_SMIT_AIDS_2026/tree/main/Assignment_12_ObjectDetection_CV), with the fixes and improvements listed below.

## 📁 Files

| File | Purpose |
|---|---|
| `app.py` | Gradio app: image detection, video tracking + export, live webcam, report viewer |
| `model_testing.ipynb` | Notebook to test the model: classes, image detection, video tracking, checkpoint comparison |
| `Pakistani_Trafic_V2.pt` | Main trained YOLO weights (~51 MB) |
| `Pakistan_Traffic_Model.pt` | Smaller/earlier checkpoint (~5 MB), faster |
| `deploy_hf.py` | Show / upload the weights on the Hugging Face Hub |
| `requirements.txt` | Python dependencies |

## 🌟 Features
- **Multi-Object Tracking**: ByteTrack or BoT-SORT keeps a unique ID for each vehicle across frames.
- **Unique counting per category**: counts distinct tracked vehicles, plus the peak number seen in one frame.
- **MP4 export at 720p / 1080p / original**, with aspect ratio preserved (letterboxed, not stretched).
- **HUD overlay** showing live counts, a per-category breakdown and FPS.
- **Webcam**: real-time frame-by-frame detection and tracking.
- **CSV reports**: a frame-by-frame detection log and a per-session summary, saved to `reports/`.

## 🚀 Run locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

To use the notebook, put a test video (`tf_test_1.mp4`) and/or image (`test_image.jpg`) in this folder and run the cells.

## 🛠️ Improvements over the original
| # | Problem in original | Fix |
|---|---|---|
| 1 | The cached model always used `persist=True`, so the **tracker state carried over between videos**: IDs kept counting up from the previous run and the tracker dropdown was ignored after the first run | Each video run gets a fresh model with a clean tracker |
| 2 | The webcam, image and video tabs **shared one model/tracker** | Each tab has its own model instance; the webcam also respects the tracker dropdown |
| 3 | The HUD bullet `•` rendered as `??` (OpenCV fonts are ASCII-only) | Replaced with `-` |
| 4 | 720p/1080p export **stretched** vertical or non-16:9 videos | Letterbox resize that keeps the aspect ratio |
| 5 | `mp4v` output often **didn't play in the browser preview** | Tries H.264 (`avc1`) first, falls back to `mp4v` |
| 6 | Videos reporting 0 frames were not processed, and the progress bar divided by zero | Reads until the end of the stream when the frame count is unknown |
| 7 | An error mid-video left the capture and writer open | `try/finally` always releases them |
| 8 | `requirements.txt` said `gradio>=4`, but the code needs Gradio 6 (`css`/`theme` passed to `launch()`) | Pinned `gradio>=6.0.0` |
| 9 | The notebook and `deploy_hf.py` hard-coded `D:\...` paths; the notebook's non-raw backslash string was also broken | Relative paths everywhere |
| 10 | The notebook's `model.track(save=True)` kept every frame in RAM (Ultralytics warned about this) | Uses `stream=True`, counts unique IDs and plots results |
| 11 | The model dropdown showed long absolute paths | Shows only the file names |
| 12 | Leftovers committed: `.ipynb_checkpoints/`, `.vscode/`, a duplicate misspelled `requirments.txt` | Removed and added to `.gitignore` |
| 13 | `deploy_hf.py` only listed files | Added an `--upload` option to push the weights |
