"""
=============================================================================
Pakistani Traffic Object Detection & Tracking System
-----------------------------------------------------------------------------
Core Technologies: Ultralytics YOLO, OpenCV, Gradio Blocks, Pandas
Model Path: <project folder>/Pakistani_Trafic_V2.pt (resolved relative to this file)
=============================================================================
"""

import os
# Prevent OpenMP runtime collision on Windows Anaconda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import gradio as gr

# ---------------------------------------------------------------------------
# Project Configuration & Directory Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Pakistani_Trafic_V2.pt")
ALT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Pakistan_Traffic_Model.pt")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
log_file_path = os.path.join(REPORTS_DIR, "system.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TrafficDetection")
logger.info("Initializing Pakistani Traffic Detection & Tracking Application...")

# ---------------------------------------------------------------------------
# Model Manager & Cache
# ---------------------------------------------------------------------------
_model_cache: Dict[Tuple[str, str], YOLO] = {}

def resolve_model_path(model_path: str) -> str:
    """Return an existing weights file, falling back to the bundled checkpoints."""
    for candidate in (model_path, DEFAULT_MODEL_PATH, ALT_MODEL_PATH):
        if candidate and os.path.exists(candidate):
            if candidate != model_path:
                logger.warning(f"Model path not found: {model_path}. Using fallback: {candidate}")
            return candidate
    raise FileNotFoundError(f"Model file not found at: {model_path}")

def get_yolo_model(model_path: str = DEFAULT_MODEL_PATH, role: str = "detect") -> YOLO:
    """
    Load and cache the YOLO model from the specified path.

    Each `role` gets its own YOLO instance, because `model.track()` stores tracker
    state on the model object. Sharing one instance between the webcam stream and
    image detection would mix up their track IDs.
    """
    model_path = resolve_model_path(model_path)
    key = (model_path, role)
    if key not in _model_cache:
        logger.info(f"Loading YOLO model ({role}) from: {model_path}")
        _model_cache[key] = YOLO(model_path)
        logger.info(f"Model successfully loaded. Classes: {_model_cache[key].names}")

    return _model_cache[key]

def letterbox_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize a frame to the target size while keeping its aspect ratio,
    padding the remaining area with black bars (no stretching of vertical videos).
    """
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def open_video_writer(path: str, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, str]:
    """
    Open an MP4 writer. H.264 ('avc1') plays directly in the browser preview;
    if this OpenCV build lacks an H.264 encoder we fall back to 'mp4v'.
    """
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return writer, "none"

# Pre-load default model
try:
    get_yolo_model(DEFAULT_MODEL_PATH)
except Exception as e:
    logger.error(f"Error during initial model load: {e}")

# ---------------------------------------------------------------------------
# HUD & Visual Annotation Utilities
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    counts: Dict[str, int],
    total_objects: int,
    fps: float = 0.0,
    title: str = "PAKISTANI TRAFFIC AI"
) -> np.ndarray:
    """
    Draw a clean, professional semi-transparent HUD overlay on the frame.
    Displays title, FPS, total detections, and category breakdown.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Panel dimensions
    card_w = min(360, int(w * 0.45))
    num_items = min(len(counts), 8)
    card_h = 75 + max(num_items * 24, 25)
    
    # Draw semi-transparent card background
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (20, 24, 33), -1)
    # Add modern accent border
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (0, 200, 115), 2)

    # Blend overlay with original frame
    alpha = 0.82
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Header title
    cv2.putText(frame, title, (28, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 150), 2, cv2.LINE_AA)
    
    # Sub-header: Total objects & FPS
    stats_str = f"Live Total: {total_objects}"
    if fps > 0:
        stats_str += f" | FPS: {fps:.1f}"
    cv2.putText(frame, stats_str, (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    
    # Divider line
    cv2.line(frame, (28, 76), (card_w, 76), (70, 80, 95), 1)

    # Category Breakdown listing
    y_pos = 98
    if not counts:
        cv2.putText(frame, "No vehicles detected", (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    else:
        for idx, (cat, cnt) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]):
            label_text = f"• {cat.capitalize()}: {cnt}"
            cv2.putText(frame, label_text, (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 245, 255), 1, cv2.LINE_AA)
            y_pos += 24

    return frame

# ---------------------------------------------------------------------------
# Core Processing: Image Detection
# ---------------------------------------------------------------------------
def process_image(
    image: np.ndarray,
    model_path: str,
    conf_threshold: float,
    iou_threshold: float
) -> Tuple[Optional[np.ndarray], pd.DataFrame, str, Optional[str]]:
    """
    Perform object detection on a single image and produce category breakdown & CSV report.
    """
    if image is None:
        return None, pd.DataFrame(), "⚠️ Please upload an image first.", None

    start_time = time.time()
    model = get_yolo_model(model_path)
    class_names = model.names

    # Convert RGB to BGR for OpenCV / YOLO processing if needed
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run inference
    results = model.predict(
        source=img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    result = results[0]
    
    # Extract detections
    counts: Dict[str, int] = {}
    records: List[Dict] = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = class_names.get(cls_id, f"Class_{cls_id}")
            conf = float(box.conf[0].item())
            xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

            counts[cls_name] = counts.get(cls_name, 0) + 1
            records.append({
                "Timestamp": timestamp_str,
                "Source_Type": "Image",
                "Detection_ID": i + 1,
                "Class_Name": cls_name,
                "Confidence": round(conf, 4),
                "BBox_X1": xyxy[0],
                "BBox_Y1": xyxy[1],
                "BBox_X2": xyxy[2],
                "BBox_Y2": xyxy[3]
            })

    total_detected = sum(counts.values())
    elapsed = time.time() - start_time

    # Generate annotated image
    annotated_bgr = result.plot()
    annotated_bgr = draw_hud(annotated_bgr, counts, total_detected, fps=(1.0 / max(elapsed, 0.001)))
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Create category breakdown dataframe
    if counts:
        df_counts = pd.DataFrame([
            {"Category": cat, "Count": cnt, "Share (%)": f"{(cnt / total_detected * 100):.1f}%"}
            for cat, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ])
    else:
        df_counts = pd.DataFrame(columns=["Category", "Count", "Share (%)"])

    # Save detailed CSV Report
    report_filename = f"image_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    if records:
        pd.DataFrame(records).to_csv(report_path, index=False)
    else:
        pd.DataFrame(columns=[
            "Timestamp", "Source_Type", "Detection_ID", "Class_Name", "Confidence",
            "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"
        ]).to_csv(report_path, index=False)

    summary_md = f"""
### 📊 Image Analysis Summary
- **Total Objects Detected:** `{total_detected}`
- **Active Categories Identified:** `{len(counts)}`
- **Inference Latency:** `{elapsed * 1000:.1f} ms`
- **Model Used:** `{os.path.basename(model_path)}`
- **Report Saved:** `{report_filename}`
"""
    return annotated_rgb, df_counts, summary_md, report_path

# ---------------------------------------------------------------------------
# Core Processing: Video Detection & Tracking with Resolution Rescaling
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model_path: str,
    conf_threshold: float,
    iou_threshold: float,
    resolution_choice: str,
    max_frames_limit: int,
    tracker_config: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], Optional[str], pd.DataFrame, str, Optional[str], Optional[str]]:
    """
    Perform real-time detection & tracking on video, scale to chosen resolution,
    encode to MP4 via OpenCV VideoWriter, and generate CSV session reports.
    """
    if not video_path or not os.path.exists(video_path):
        return None, None, pd.DataFrame(), "⚠️ Please provide a valid input video file.", None, None

    logger.info(f"Starting video processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, pd.DataFrame(), "❌ Error: Could not open the specified video file.", None, None

    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or np.isnan(source_fps):
        source_fps = 30.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolve target output resolution
    resolution_map = {
        "720p (1280x720)": (1280, 720),
        "1080p (1920x1080)": (1920, 1080),
        "Original": (orig_w, orig_h)
    }
    target_w, target_h = resolution_map.get(resolution_choice, (1280, 720))

    # Number of frames to process
    frames_to_process = total_source_frames
    if max_frames_limit > 0:
        frames_to_process = min(total_source_frames, max_frames_limit)

    # Configure output file path
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_filename = f"tracked_{resolution_choice.split()[0]}_{timestamp_prefix}.mp4"
    out_video_path = os.path.join(OUTPUTS_DIR, out_video_filename)

    # Initialize OpenCV VideoWriter using 'mp4v' codec as required
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, source_fps, (target_w, target_h))

    if not out_writer.isOpened():
        cap.release()
        return None, None, pd.DataFrame(), f"❌ Failed to create video writer at: {out_video_path}", None, None

    model = get_yolo_model(model_path)
    class_names = model.names

    # Data collection for reports & metrics
    frame_idx = 0
    all_detections_log: List[Dict] = []
    unique_tracked_per_class: Dict[str, set] = {}
    current_frame_counts: Dict[str, int] = {}
    total_detections_count = 0
    start_time = time.time()

    progress(0.0, desc="Starting Video Tracking Pipeline...")

    while cap.isOpened() and frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        t_frame_start = time.time()

        # Run YOLO Tracking with persistent tracking across frames
        results = model.track(
            source=frame,
            persist=True,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker=tracker_config,
            verbose=False
        )
        res = results[0]

        # Reset current frame counts
        current_frame_counts.clear()

        # Process detected & tracked bounding boxes
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = class_names.get(cls_id, f"Class_{cls_id}")
                conf = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else None
                xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                current_frame_counts[cls_name] = current_frame_counts.get(cls_name, 0) + 1
                total_detections_count += 1

                # Track unique object IDs
                if cls_name not in unique_tracked_per_class:
                    unique_tracked_per_class[cls_name] = set()
                if track_id is not None:
                    unique_tracked_per_class[cls_name].add(track_id)

                all_detections_log.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Frame_Number": frame_idx,
                    "Time_Seconds": round(frame_idx / source_fps, 2),
                    "Track_ID": track_id if track_id is not None else "N/A",
                    "Class_Name": cls_name,
                    "Confidence": round(conf, 4),
                    "BBox_X1": xyxy[0],
                    "BBox_Y1": xyxy[1],
                    "BBox_X2": xyxy[2],
                    "BBox_Y2": xyxy[3]
                })

        # Draw YOLO annotations (bounding boxes & tracks)
        annotated_frame = res.plot()

        # Calculate current frame FPS
        frame_fps = 1.0 / max(time.time() - t_frame_start, 0.001)

        # Draw Professional HUD banner
        annotated_frame = draw_hud(
            annotated_frame,
            current_frame_counts,
            total_objects=sum(current_frame_counts.values()),
            fps=frame_fps,
            title="PAKISTANI TRAFFIC TRACKER"
        )

        # Scale to selected target resolution
        if (annotated_frame.shape[1], annotated_frame.shape[0]) != (target_w, target_h):
            annotated_frame = cv2.resize(annotated_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Write frame to output video
        out_writer.write(annotated_frame)

        # Update progress bar
        if frame_idx % 5 == 0 or frame_idx == frames_to_process:
            pct = frame_idx / frames_to_process
            progress(pct, desc=f"Tracking Frame {frame_idx}/{frames_to_process} ({pct*100:.1f}%)")

    # Cleanup resources
    cap.release()
    out_writer.release()
    total_time_taken = time.time() - start_time
    avg_fps = frame_idx / max(total_time_taken, 0.001)

    logger.info(f"Video processing finished. Processed {frame_idx} frames in {total_time_taken:.2f}s ({avg_fps:.1f} FPS).")

    # Generate Category Breakdown DataFrame
    category_summary_rows = []
    total_unique_objects = sum(len(ids) for ids in unique_tracked_per_class.values())

    for cls_name, id_set in sorted(unique_tracked_per_class.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(id_set)
        share = f"{(count / max(total_unique_objects, 1) * 100):.1f}%" if total_unique_objects > 0 else "0.0%"
        category_summary_rows.append({
            "Category": cls_name,
            "Unique Tracked Objects": count,
            "Share (%)": share
        })

    if not category_summary_rows:
        df_category_breakdown = pd.DataFrame(columns=["Category", "Unique Tracked Objects", "Share (%)"])
    else:
        df_category_breakdown = pd.DataFrame(category_summary_rows)

    # Export Detailed Session Log to CSV
    detailed_csv_filename = f"traffic_video_log_{timestamp_prefix}.csv"
    detailed_csv_path = os.path.join(REPORTS_DIR, detailed_csv_filename)
    if all_detections_log:
        pd.DataFrame(all_detections_log).to_csv(detailed_csv_path, index=False)
    else:
        pd.DataFrame(columns=[
            "Timestamp", "Frame_Number", "Time_Seconds", "Track_ID", "Class_Name",
            "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"
        ]).to_csv(detailed_csv_path, index=False)

    # Export Summary CSV
    summary_csv_filename = f"traffic_summary_{timestamp_prefix}.csv"
    summary_csv_path = os.path.join(REPORTS_DIR, summary_csv_filename)
    df_category_breakdown.to_csv(summary_csv_path, index=False)

    # Format Summary Markdown for UI
    summary_md = f"""
### 🎬 Video Processing & Tracking Completed!
- **Target Resolution:** `{target_w}x{target_h}` ({resolution_choice})
- **Frames Processed:** `{frame_idx}` / `{total_source_frames}`
- **Average Processing Speed:** `{avg_fps:.1f} FPS` (Total Time: `{total_time_taken:.1f}s`)
- **Total Unique Tracked Vehicles:** `{total_unique_objects}`
- **Active Traffic Categories:** `{len(unique_tracked_per_class)}`
- **Output Video Saved At:** `{out_video_filename}`
"""
    return out_video_path, out_video_path, df_category_breakdown, summary_md, detailed_csv_path, summary_csv_path

# ---------------------------------------------------------------------------
# Core Processing: Live Webcam Frame Stream
# ---------------------------------------------------------------------------
_webcam_tracker_state = {"last_time": time.time(), "fps": 0.0}

def process_webcam_frame(
    frame: np.ndarray,
    model_path: str,
    conf_threshold: float,
    iou_threshold: float
) -> Tuple[Optional[np.ndarray], pd.DataFrame, str]:
    """
    Process incoming real-time webcam frame with detection, tracking, HUD overlay,
    and dynamic category counts.
    """
    if frame is None:
        return None, pd.DataFrame(), "Webcam inactive."

    now = time.time()
    dt = now - _webcam_tracker_state["last_time"]
    _webcam_tracker_state["last_time"] = now
    fps = 1.0 / max(dt, 0.001)
    _webcam_tracker_state["fps"] = 0.8 * _webcam_tracker_state["fps"] + 0.2 * fps

    model = get_yolo_model(model_path)
    class_names = model.names

    # Convert RGB frame to BGR for OpenCV / YOLO
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run YOLO tracking on live frame
    results = model.track(
        source=frame_bgr,
        persist=True,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    res = results[0]

    counts: Dict[str, int] = {}
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = class_names.get(cls_id, f"Class_{cls_id}")
            counts[cls_name] = counts.get(cls_name, 0) + 1

    total_detected = sum(counts.values())

    # Plot bounding boxes & track IDs
    annotated_bgr = res.plot()
    # Draw professional HUD
    annotated_bgr = draw_hud(
        annotated_bgr,
        counts,
        total_objects=total_detected,
        fps=_webcam_tracker_state["fps"],
        title="LIVE WEBCAM CV MONITOR"
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Category counts table
    if counts:
        df_live = pd.DataFrame([
            {"Category": k, "Live Count": v}
            for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ])
    else:
        df_live = pd.DataFrame(columns=["Category", "Live Count"])

    status_md = f"**Status:** Streaming | **Live Detections:** {total_detected} | **FPS:** {_webcam_tracker_state['fps']:.1f}"
    return annotated_rgb, df_live, status_md

# ---------------------------------------------------------------------------
# Reports Management & Session Log Viewer
# ---------------------------------------------------------------------------
def refresh_reports_list() -> List[str]:
    """List all CSV session reports available in the reports directory."""
    if not os.path.exists(REPORTS_DIR):
        return []
    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".csv") or f.endswith(".log")]
    files.sort(reverse=True)
    return [os.path.join(REPORTS_DIR, f) for f in files]

def load_selected_report(file_path: str) -> Tuple[pd.DataFrame, str]:
    """Load and preview a selected CSV log report."""
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame(), "Please select a valid report file."
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            info = f"Loaded `{os.path.basename(file_path)}` ({len(df)} rows, {len(df.columns)} columns)."
            return df, info
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()[-2000:]
            return pd.DataFrame([{"Log Content": content}]), f"Log tail for `{os.path.basename(file_path)}`"
    except Exception as e:
        return pd.DataFrame(), f"Error loading file: {str(e)}"

# ---------------------------------------------------------------------------
# Gradio UI Theme & Custom Styling
# ---------------------------------------------------------------------------
custom_css = """
/* Modern Dark Glassmorphic Dashboard Theme */
body, .gradio-container {
    background-color: #0f141c !important;
    color: #e5e9f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.main-header {
    background: linear-gradient(135deg, #16202c 0%, #1e2c3d 50%, #17324d 100%);
    border: 1px solid #2e4359;
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.main-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00e699;
    letter-spacing: -0.5px;
}

.main-header p {
    margin: 8px 0 0 0;
    color: #a0b2c6;
    font-size: 1.05rem;
}

.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(0, 230, 153, 0.15);
    color: #00e699;
    border: 1px solid rgba(0, 230, 153, 0.3);
    margin-right: 8px;
}

.card-panel {
    background: #151b24 !important;
    border: 1px solid #232f3e !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

.primary-btn {
    background: linear-gradient(135deg, #00b377 0%, #008f5d 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
}

.primary-btn:hover {
    background: linear-gradient(135deg, #00d68f 0%, #00a86b 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0, 230, 153, 0.3) !important;
}

.stats-box {
    background: #182230;
    border-left: 4px solid #00e699;
    padding: 12px 18px;
    border-radius: 6px;
    margin-top: 12px;
}
"""

# Available models
model_choices = [
    DEFAULT_MODEL_PATH,
    ALT_MODEL_PATH
]
# Filter to existing files
existing_model_choices = [m for m in model_choices if os.path.exists(m)]
if not existing_model_choices:
    existing_model_choices = [DEFAULT_MODEL_PATH]

# ---------------------------------------------------------------------------
# Construct Gradio Blocks Application
# ---------------------------------------------------------------------------
with gr.Blocks(title="Pakistani Traffic AI - YOLO Object Detection & Tracking") as demo:
    # Header Banner
    gr.HTML("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1>🚦 Pakistani Traffic AI - Object Detection & Tracking</h1>
                <p>Enterprise Computer Vision System powered by custom YOLO with real-time multi-object tracking, category counting, and analytical reporting.</p>
            </div>
            <div style="margin-top: 10px;">
                <span class="badge-pill">⚡ Ultralytics YOLO</span>
                <span class="badge-pill">🎯 ByteTrack / BoT-SORT</span>
                <span class="badge-pill">📊 Real-Time Analytics</span>
            </div>
        </div>
    </div>
    """)

    # Global Settings Bar
    with gr.Accordion("⚙️ Global Model & Inference Settings", open=False):
        with gr.Row():
            global_model_dropdown = gr.Dropdown(
                choices=existing_model_choices,
                value=existing_model_choices[0],
                label="Select YOLO Model Weights (.pt)",
                info="Default: Pakistani_Trafic_V2.pt (or custom checkpoint)"
            )
            global_tracker_dropdown = gr.Dropdown(
                choices=["bytetrack.yaml", "botsort.yaml"],
                value="bytetrack.yaml",
                label="Multi-Object Tracker Algorithm",
                info="ByteTrack is recommended for fast vehicle tracking"
            )

    # Main Navigation Tabs
    with gr.Tabs():
        # ===================================================================
        # TAB 1: Video Detection & Tracking
        # ===================================================================
        with gr.Tab("🎬 Video Detection & Tracking", id="tab_video"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=5):
                    video_input = gr.Video(
                        label="Upload Traffic Video (MP4, AVI, MOV)",
                        sources=["upload"]
                    )
                    
                    with gr.Row():
                        video_resolution_choice = gr.Radio(
                            choices=["720p (1280x720)", "1080p (1920x1080)", "Original"],
                            value="720p (1280x720)",
                            label="Output Export Resolution (OpenCV mp4v)",
                            info="Select desired target resolution for exported video"
                        )
                    
                    with gr.Row():
                        video_conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        video_iou_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                            label="IoU / NMS Threshold"
                        )

                    with gr.Row():
                        video_max_frames = gr.Slider(
                            minimum=0, maximum=2000, value=250, step=50,
                            label="Max Frames to Process (0 = Entire Video)",
                            info="Set to 200-300 frames for quick testing or 0 for entire video"
                        )

                    video_process_btn = gr.Button(
                        "🚀 Start Detection & Tracking",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )

                # Right Column: Outputs, Visuals & Analytics
                with gr.Column(scale=6):
                    video_output_player = gr.Video(
                        label="🎥 Processed & Tracked Video Preview",
                        interactive=False
                    )

                    with gr.Row():
                        video_download_file = gr.File(
                            label="📥 Download Processed Video (MP4)"
                        )
                        video_csv_download = gr.File(
                            label="📥 Download Detailed CSV Log"
                        )
                        video_summary_download = gr.File(
                            label="📥 Download Summary CSV"
                        )

                    video_counts_df = gr.DataFrame(
                        headers=["Category", "Unique Tracked Objects", "Share (%)"],
                        label="📊 Category-Wise Vehicle Breakdown",
                        interactive=False
                    )

                    video_summary_md = gr.Markdown(
                        "Upload a video and click **Start Detection & Tracking** to view metrics."
                    )

            # Wire Video Processing Event
            video_process_btn.click(
                fn=process_video,
                inputs=[
                    video_input,
                    global_model_dropdown,
                    video_conf_slider,
                    video_iou_slider,
                    video_resolution_choice,
                    video_max_frames,
                    global_tracker_dropdown
                ],
                outputs=[
                    video_output_player,
                    video_download_file,
                    video_counts_df,
                    video_summary_md,
                    video_csv_download,
                    video_summary_download
                ]
            )

        # ===================================================================
        # TAB 2: Image Detection & Analysis
        # ===================================================================
        with gr.Tab("📷 Image Detection & Analysis", id="tab_image"):
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(
                        label="Upload Traffic Image",
                        type="numpy"
                    )
                    with gr.Row():
                        image_conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        image_iou_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                            label="IoU / NMS Threshold"
                        )
                    image_process_btn = gr.Button(
                        "🔍 Detect Objects",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )

                with gr.Column(scale=6):
                    image_output = gr.Image(
                        label="Annotated Image with HUD",
                        type="numpy"
                    )
                    image_counts_df = gr.DataFrame(
                        headers=["Category", "Count", "Share (%)"],
                        label="📊 Detected Categories Breakdown",
                        interactive=False
                    )
                    image_summary_md = gr.Markdown("Detection details will appear here.")
                    image_csv_download = gr.File(
                        label="📥 Download Image Detection CSV Report"
                    )

            image_process_btn.click(
                fn=process_image,
                inputs=[
                    image_input,
                    global_model_dropdown,
                    image_conf_slider,
                    image_iou_slider
                ],
                outputs=[
                    image_output,
                    image_counts_df,
                    image_summary_md,
                    image_csv_download
                ]
            )

        # ===================================================================
        # TAB 3: Real-Time Webcam Stream
        # ===================================================================
        with gr.Tab("📹 Real-Time Webcam Feed", id="tab_webcam"):
            gr.Markdown("""
            ### 🎥 Live Webcam Traffic Monitoring
            Connect your camera to stream frames in real-time. The YOLO model tracks objects frame-by-frame,
            dynamically overlays bounding boxes, tracking IDs, and updates live vehicle counters.
            """)
            with gr.Row():
                with gr.Column(scale=6):
                    webcam_stream_in = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        label="Live Webcam Input Stream"
                    )
                    with gr.Row():
                        webcam_conf_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.30, step=0.05,
                            label="Confidence Threshold"
                        )
                        webcam_iou_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                            label="IoU Threshold"
                        )

                with gr.Column(scale=6):
                    webcam_stream_out = gr.Image(
                        label="Processed Live Feed with HUD & Trackers",
                        type="numpy"
                    )
                    webcam_counts_df = gr.DataFrame(
                        headers=["Category", "Live Count"],
                        label="🔴 Live Category Counter",
                        interactive=False
                    )
                    webcam_status_md = gr.Markdown("**Status:** Ready. Start camera streaming above.")

            # Real-time streaming connection
            webcam_stream_in.stream(
                fn=process_webcam_frame,
                inputs=[
                    webcam_stream_in,
                    global_model_dropdown,
                    webcam_conf_slider,
                    webcam_iou_slider
                ],
                outputs=[
                    webcam_stream_out,
                    webcam_counts_df,
                    webcam_status_md
                ]
            )

        # ===================================================================
        # TAB 4: Session Analytics & Log Viewer
        # ===================================================================
        with gr.Tab("📊 Session Analytics & Reports", id="tab_logs"):
            gr.Markdown("""
            ### 📁 Historical Detection Reports & Logs
            Every detection and tracking session automatically generates an audit log in the project `reports/` folder.
            Select any report below to inspect the data or download it.
            """)
            with gr.Row():
                with gr.Column(scale=4):
                    reports_dropdown = gr.Dropdown(
                        choices=refresh_reports_list(),
                        label="Available Report Logs (CSV / Log)",
                        interactive=True
                    )
                    refresh_reports_btn = gr.Button("🔄 Refresh Reports List")
                    download_selected_report_btn = gr.File(label="📥 Download Selected Report")

                with gr.Column(scale=8):
                    report_info_md = gr.Markdown("Select a report from the dropdown to preview.")
                    report_dataframe_view = gr.DataFrame(
                        label="Log Preview Table",
                        interactive=False
                    )

            refresh_reports_btn.click(
                fn=lambda: gr.update(choices=refresh_reports_list()),
                inputs=[],
                outputs=[reports_dropdown]
            )

            reports_dropdown.change(
                fn=load_selected_report,
                inputs=[reports_dropdown],
                outputs=[report_dataframe_view, report_info_md]
            ).then(
                fn=lambda path: path,
                inputs=[reports_dropdown],
                outputs=[download_selected_report_btn]
            )

    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 15px; color: #6c7d93; font-size: 0.88rem; border-top: 1px solid #1c2633;">
        Pakistani Traffic AI System • Developed with Ultralytics YOLO & Gradio • Model Checkpoint: Pakistani_Trafic_V2.pt
    </div>
    """)

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Support both local environment and Hugging Face Spaces container environment
    is_hf_space = bool(os.environ.get("SPACE_ID"))
    server_name = "0.0.0.0" if is_hf_space else "127.0.0.1"
    server_port = int(os.environ.get("PORT", 7860))

    logger.info(f"Launching Gradio Traffic Detection App on {server_name}:{server_port} ...")
    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css
    )
