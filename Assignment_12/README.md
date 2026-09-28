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

# 🚦 Pakistani Traffic Object Detection & Multi-Object Tracking

An enterprise Computer Vision application developed with **Ultralytics YOLO** and **Gradio** specifically trained and tuned for Pakistani traffic dynamics (Rickshaws, Bikes, Buses, Cars, and Pedestrians).

## 🌟 Key Features
- **Real-Time Multi-Object Tracking (MOT)**: Leverages ByteTrack and BoT-SORT algorithms to maintain unique vehicle track IDs across frames.
- **Dynamic Category Breakdown**: Live counting of detected vehicles by specific category.
- **Resolution Scaling & MP4 Export**: Export processed videos in 720p, 1080p, or original resolution using OpenCV `mp4v` codec.
- **HUD Telemetry Overlay**: Professional Head-Up Display directly rendered on frames showing live counts, category metrics, and FPS.
- **Webcam Support**: Real-time webcam frame streaming and processing.
- **Audit & Session CSV Reports**: Automatically generates and downloads detailed frame-by-frame logs and summary reports.

## 🚀 Running Locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:7860` in your web browser.
