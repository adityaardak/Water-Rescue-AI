import hashlib
import inspect
import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


APP_TITLE = "Water Rescue AI"
SEA_OBJECT_LABELS = [
    "lifebuoy",
    "rescue tube",
    "rope",
    "boat",
    "jet ski",
    "ladder",
    "pool edge",
    "floating device",
    "rescue board",
    "buoy",
    "surfboard",
    "kayak",
    "canoe",
    "dock",
    "pier",
    "shore",
    "rock",
    "debris",
]
NO_RESCUE_ALERT = "No nearby object identified. Rescue operation may be required."
DISTANT_RESCUE_ALERT = "Nearest object appears distant. Immediate attention recommended."
NEARBY_RESCUE_ALERT = (
    "Nearest object identified. Rescue support may be possible through this object."
)


@dataclass
class Detection:
    label: str
    confidence: float
    box: List[float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def set_page_config() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --navy: #071b33;
                --blue: #0b5d8f;
                --aqua: #19c7d4;
                --ink: #112235;
                --muted: #607487;
                --danger: #c72534;
                --warning: #bc7a00;
                --success: #087f5b;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(25, 199, 212, 0.16), transparent 32rem),
                    linear-gradient(180deg, #f5fbff 0%, #eef5f9 52%, #ffffff 100%);
                color: var(--ink);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1220px;
            }
            [data-testid="stSidebar"] {
                background: #061a30;
            }
            [data-testid="stSidebar"] * {
                color: #f4fbff;
            }
            [data-testid="stSidebar"] .stSlider p,
            [data-testid="stSidebar"] .stTextArea p,
            [data-testid="stSidebar"] .stCheckbox p,
            [data-testid="stSidebar"] .stNumberInput p {
                color: #d9edf7;
            }
            .hero {
                padding: 2rem;
                border-radius: 8px;
                background: linear-gradient(135deg, #061a30 0%, #0b5d8f 62%, #11bfcf 100%);
                box-shadow: 0 18px 42px rgba(7, 27, 51, 0.20);
                color: #ffffff;
                margin-bottom: 1.35rem;
            }
            .hero h1 {
                font-size: 3rem;
                line-height: 1.05;
                letter-spacing: 0;
                margin: 0 0 .55rem 0;
            }
            .hero p {
                font-size: 1.08rem;
                color: #eaf9ff;
                margin: .2rem 0;
                max-width: 760px;
            }
            .section-card {
                border-bottom: 1px solid rgba(8, 79, 120, 0.10);
                padding: 1rem 0 1.2rem 0;
                margin: .6rem 0 1rem 0;
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid rgba(8, 79, 120, 0.12);
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(7, 27, 51, 0.07);
                padding: .95rem 1rem;
                min-height: 104px;
            }
            .metric-label {
                color: var(--muted);
                font-size: .85rem;
                margin-bottom: .35rem;
            }
            .metric-value {
                color: var(--navy);
                font-weight: 800;
                font-size: 1.35rem;
                overflow-wrap: anywhere;
            }
            .badge {
                display: inline-flex;
                align-items: center;
                gap: .35rem;
                border-radius: 8px;
                padding: .48rem .68rem;
                font-weight: 700;
                margin: .25rem .25rem .25rem 0;
            }
            .badge.info { background: #e7f7ff; color: #075985; }
            .badge.good { background: #e5f7ef; color: var(--success); }
            .badge.warn { background: #fff4dc; color: var(--warning); }
            .badge.danger { background: #fde7ea; color: var(--danger); }
            .footer {
                color: #486174;
                text-align: center;
                padding: 1.5rem 0 .5rem;
                font-weight: 600;
            }
            .stButton > button {
                border-radius: 8px;
                min-height: 3rem;
                font-weight: 800;
                border: 1px solid #0b7ea8;
                box-shadow: 0 10px 22px rgba(11, 93, 143, 0.18);
            }
            .stDownloadButton > button {
                border-radius: 8px;
            }
            @media (max-width: 760px) {
                .hero { padding: 1.25rem; }
                .hero h1 { font-size: 2.2rem; }
                .section-card { padding: 1rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🌊 Water Rescue AI</h1>
            <p><strong>Detect a person in water, find nearby sea objects, and support faster rescue decisions.</strong></p>
            <p>Upload a pool, sea, river, lake, or flood scene for calm visual triage and clearer rescue awareness.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(message: str, style: str = "info") -> None:
    st.markdown(f'<span class="badge {style}">{message}</span>', unsafe_allow_html=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_yolo_model() -> YOLO:
    # YOLO11n keeps the app practical for student laptops while still using YOLO11.
    return YOLO("yolo11n.pt")


@st.cache_resource(show_spinner=False)
def load_grounding_dino_model() -> Tuple[Any, Any, torch.device]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = get_device()
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_zoe_depth_model() -> Tuple[Any, Any, torch.device, str]:
    device = get_device()
    model_id = "Intel/zoedepth-nyu-kitti"
    try:
        from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = ZoeDepthForDepthEstimation.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return processor, model, device, "native"
    except Exception:
        from transformers import pipeline

        pipe_device = 0 if torch.cuda.is_available() else -1
        depth_pipe = pipeline("depth-estimation", model=model_id, device=pipe_device)
        return depth_pipe, None, device, "pipeline"


def file_signature(file_bytes: bytes, mode: str, *settings: Any) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    setting_blob = "|".join(str(item) for item in settings)
    return f"{mode}:{digest}:{setting_blob}"


def sea_object_labels() -> List[str]:
    return SEA_OBJECT_LABELS.copy()


def read_uploaded_image(file_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes))
    return image.convert("RGB")


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_video_frames(
    file_bytes: bytes,
    sampling_interval_seconds: int,
    max_frames: int,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        video_path = tmp.name

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise ValueError("This video could not be opened. Please try a valid MP4, MOV, or AVI file.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        # Sampling keeps video analysis responsive and avoids running models on every frame.
        step = max(1, int(round(fps * sampling_interval_seconds)))
        frame_index = 0
        while len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % step == 0:
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(frame_index / fps, 2),
                        "image": bgr_to_pil(frame),
                    }
                )
            frame_index += 1
    finally:
        capture.release()
        try:
            os.remove(video_path)
        except OSError:
            pass

    if not frames:
        raise ValueError("No usable frames were found in this video.")
    return frames


def detect_people(image: Image.Image, confidence_threshold: float) -> List[Detection]:
    model = load_yolo_model()
    results = model.predict(
        np.array(image),
        conf=confidence_threshold,
        classes=[0],
        verbose=False,
    )
    detections: List[Detection] = []
    if not results or results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
        confidence = float(box.conf[0].detach().cpu().item())
        detections.append(Detection(label="person", confidence=confidence, box=xyxy))
    return detections


def draw_detections(
    image: Image.Image,
    people: Optional[List[Detection]] = None,
    objects: Optional[List[Detection]] = None,
    show_boxes: bool = True,
) -> Image.Image:
    canvas = image.copy()
    if not show_boxes:
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    def draw_one(detection: Detection, color: str, prefix: str) -> None:
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{prefix} {detection.label} {detection.confidence:.2f}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_box[0] - 3, text_box[1] - 3, text_box[2] + 3, text_box[3] + 3],
            fill=color,
        )
        draw.text((x1, y1), label, fill="white", font=font)

    for detection in people or []:
        draw_one(detection, "#0b5d8f", "person")
    for detection in objects or []:
        draw_one(detection, "#08a66c", "rescue")
    return canvas


def average_confidence(detections: List[Detection]) -> float:
    if not detections:
        return 0.0
    return float(np.mean([d.confidence for d in detections]))


def analyze_frames_for_people(
    frames: List[Dict[str, Any]],
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    analyzed = []
    for frame in frames:
        detections = detect_people(frame["image"], confidence_threshold)
        analyzed.append({**frame, "people": detections})
    return analyzed


def build_initial_event_log(analyzed_frames: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, frame in enumerate(analyzed_frames):
        people = frame["people"]
        count = len(people)
        if count == 0:
            alert_level = "Info"
            alert_message = "No person detected in water scene."
        elif count == 1:
            alert_level = "Warning"
            alert_message = "Potential rescue attention required."
        else:
            alert_level = "Critical"
            alert_message = "Multiple people detected. Potential rescue attention required."

        rows.append(
            {
                "frame_id": index,
                "timestamp": frame.get("timestamp", 0.0),
                "person_detected": "Yes" if count else "No",
                "number_people": count,
                "person_confidence": round(average_confidence(people), 3),
                "rescue_object_status": "Not searched",
                "nearest_rescue_object": "",
                "estimated_closeness": "",
                "alert_level": alert_level,
                "alert_message": alert_message,
            }
        )
    return pd.DataFrame(rows)


def choose_critical_frame(analyzed_frames: List[Dict[str, Any]]) -> int:
    scored = []
    for index, frame in enumerate(analyzed_frames):
        people = frame["people"]
        scored.append((len(people), average_confidence(people), index))
    scored.sort(reverse=True)
    return scored[0][2] if scored else 0


def detect_rescue_objects(
    image: Image.Image,
    prompt_labels: List[str],
    confidence_threshold: float,
) -> List[Detection]:
    processor, model, device = load_grounding_dino_model()
    prompt = ". ".join(prompt_labels) + "."
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    text_threshold = max(0.15, min(0.35, confidence_threshold))
    post_process = processor.post_process_grounded_object_detection
    post_process_params = inspect.signature(post_process).parameters
    post_kwargs = {
        "outputs": outputs,
        "input_ids": inputs.get("input_ids"),
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    # Transformers changed this argument from box_threshold to threshold.
    if "box_threshold" in post_process_params:
        post_kwargs["box_threshold"] = confidence_threshold
    else:
        post_kwargs["threshold"] = confidence_threshold
    if "text_labels" in post_process_params:
        post_kwargs["text_labels"] = [prompt_labels]

    processed = post_process(**post_kwargs)[0]

    boxes = processed.get("boxes", [])
    scores = processed.get("scores", [])
    labels = processed.get("labels", ["rescue object"] * len(boxes))

    detections: List[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        score_value = float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)
        box_values = box.detach().cpu().numpy().astype(float).tolist() if hasattr(box, "detach") else list(box)
        label_text = str(label).strip() or "rescue object"
        detections.append(Detection(label=label_text, confidence=score_value, box=box_values))
    return detections


def estimate_depth_map(image: Image.Image) -> np.ndarray:
    processor, model, device, mode = load_zoe_depth_model()
    if mode == "native":
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        post_processed = processor.post_process_depth_estimation(
            outputs,
            source_sizes=[(image.height, image.width)],
        )
        depth = post_processed[0]["predicted_depth"]
        depth_array = depth.detach().cpu().numpy()
    else:
        result = processor(image)
        if "predicted_depth" in result:
            predicted = result["predicted_depth"]
            depth_array = predicted.detach().cpu().numpy() if hasattr(predicted, "detach") else np.array(predicted)
        else:
            depth_array = np.array(result["depth"]).astype(np.float32)

    depth_array = np.squeeze(depth_array).astype(np.float32)
    if depth_array.shape[:2] != (image.height, image.width):
        depth_array = cv2.resize(depth_array, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
    return depth_array


def depth_to_preview(depth_array: np.ndarray) -> Image.Image:
    clean = np.nan_to_num(depth_array, nan=np.nanmedian(depth_array))
    min_depth = float(np.min(clean))
    max_depth = float(np.max(clean))
    if math.isclose(min_depth, max_depth):
        normalized = np.zeros_like(clean, dtype=np.uint8)
    else:
        normalized = ((clean - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return bgr_to_pil(colored)


def median_depth_in_box(depth_array: np.ndarray, box: List[float]) -> float:
    height, width = depth_array.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    crop = depth_array[y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)]
    return float(np.nanmedian(crop)) if crop.size else float(np.nanmedian(depth_array))


def calculate_object_distances(
    people: List[Detection],
    rescue_objects: List[Detection],
    depth_array: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if not people or not rescue_objects:
        return []

    width, height = image_size
    diagonal = math.sqrt(width * width + height * height)
    depth_range = 1.0
    if depth_array is not None:
        clean_depth = np.nan_to_num(depth_array, nan=np.nanmedian(depth_array))
        depth_range = max(1e-6, float(np.nanpercentile(clean_depth, 95) - np.nanpercentile(clean_depth, 5)))

    candidates = []
    for person in people:
        person_depth = median_depth_in_box(depth_array, person.box) if depth_array is not None else 0.0
        px, py = person.center
        for obj in rescue_objects:
            ox, oy = obj.center
            pixel_distance = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            spatial_score = min(1.0, pixel_distance / diagonal)
            object_depth = median_depth_in_box(depth_array, obj.box) if depth_array is not None else person_depth
            depth_score = min(1.0, abs(object_depth - person_depth) / depth_range)
            # This is a relative rescue closeness score, not a physical meter estimate.
            distance_score = (0.68 * spatial_score) + (0.32 * depth_score)
            closeness = round(max(0.0, min(1.0, 1.0 - distance_score)), 3)
            candidates.append(
                {
                    "person": person,
                    "object": obj,
                    "closeness": closeness,
                    "distance_score": round(distance_score, 3),
                    "confidence_level": round((obj.confidence * 0.72) + (closeness * 0.28), 3),
                }
            )
    return sorted(candidates, key=lambda item: (item["closeness"], item["object"].confidence), reverse=True)


def find_nearest_rescue_object(
    people: List[Detection],
    rescue_objects: List[Detection],
    depth_array: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    candidates = calculate_object_distances(people, rescue_objects, depth_array, image_size)
    return candidates[0] if candidates else None


def alert_from_nearest(nearest: Optional[Dict[str, Any]], rescue_objects: List[Detection]) -> Tuple[str, str, str]:
    if not rescue_objects or nearest is None:
        return "Critical", "None found", NO_RESCUE_ALERT
    if nearest["closeness"] < 0.48:
        return "Warning", "Found distant", DISTANT_RESCUE_ALERT
    return "Good", "Found nearby", NEARBY_RESCUE_ALERT


def run_rescue_analysis(
    analyzed_frames: List[Dict[str, Any]],
    event_log: pd.DataFrame,
    prompt_labels: List[str],
    object_confidence: float,
) -> Dict[str, Any]:
    selected_index = choose_critical_frame(analyzed_frames)
    selected = analyzed_frames[selected_index]

    rescue_objects = detect_rescue_objects(selected["image"], prompt_labels, object_confidence)
    depth_array = estimate_depth_map(selected["image"])
    object_distances = calculate_object_distances(
        selected["people"],
        rescue_objects,
        depth_array,
        selected["image"].size,
    )
    nearest = object_distances[0] if object_distances else None
    alert_level, rescue_status, alert_message = alert_from_nearest(nearest, rescue_objects)

    updated_log = event_log.copy()
    updated_log.loc[selected_index, "rescue_object_status"] = rescue_status
    updated_log.loc[selected_index, "nearest_rescue_object"] = nearest["object"].label if nearest else ""
    updated_log.loc[selected_index, "estimated_closeness"] = f"{nearest['closeness']:.3f}" if nearest else ""
    updated_log.loc[selected_index, "alert_level"] = alert_level
    updated_log.loc[selected_index, "alert_message"] = alert_message

    annotated = draw_detections(selected["image"], selected["people"], rescue_objects, True)
    return {
        "selected_index": selected_index,
        "selected_timestamp": selected.get("timestamp", 0.0),
        "objects": rescue_objects,
        "object_distances": object_distances,
        "nearest": nearest,
        "depth": depth_array,
        "depth_preview": depth_to_preview(depth_array),
        "annotated": annotated,
        "event_log": updated_log,
        "alert_level": alert_level,
        "alert_message": alert_message,
    }


def summarize_events(event_log: pd.DataFrame) -> Dict[str, Any]:
    searched = event_log[event_log["rescue_object_status"] != "Not searched"]
    critical = event_log[event_log["alert_level"] == "Critical"]
    nearest_found = bool((event_log["nearest_rescue_object"].astype(str).str.len() > 0).any())
    most_recent = event_log.iloc[-1]["alert_message"] if not event_log.empty else "No analysis yet."
    if not searched.empty:
        most_recent = searched.iloc[-1]["alert_message"]

    return {
        "total_frames_analyzed": int(len(event_log)),
        "total_person_detections": int(event_log["number_people"].sum()) if not event_log.empty else 0,
        "total_rescue_object_searches": int(len(searched)),
        "nearest_object_found": "Yes" if nearest_found else "No",
        "critical_alert_count": int(len(critical)),
        "most_recent_alert": most_recent,
    }


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=18, r=18, t=28, b=18),
        font=dict(color="#112235"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_person_count_chart(event_log: pd.DataFrame):
    chart_df = event_log.copy()
    chart_df["timestamp_label"] = chart_df["timestamp"].astype(str) + "s"
    return px.bar(
        chart_df,
        x="timestamp_label",
        y="number_people",
        labels={"timestamp_label": "Timestamp", "number_people": "People detected"},
        color_discrete_sequence=["#0b5d8f"],
    )


def create_rescue_count_chart(event_log: pd.DataFrame):
    chart_df = event_log.copy()
    chart_df["rescue_count"] = np.where(chart_df["nearest_rescue_object"].astype(str).str.len() > 0, 1, 0)
    chart_df["timestamp_label"] = chart_df["timestamp"].astype(str) + "s"
    return px.bar(
        chart_df,
        x="timestamp_label",
        y="rescue_count",
        labels={"timestamp_label": "Timestamp", "rescue_count": "Nearest object found"},
        color_discrete_sequence=["#19c7d4"],
    )


def create_alert_chart(event_log: pd.DataFrame):
    counts = event_log["alert_level"].value_counts().rename_axis("alert_level").reset_index(name="count")
    return px.pie(
        counts,
        names="alert_level",
        values="count",
        color="alert_level",
        color_discrete_map={
            "Info": "#8aa4b8",
            "Good": "#087f5b",
            "Warning": "#bc7a00",
            "Critical": "#c72534",
        },
        hole=0.45,
    )


def create_timeline_chart(event_log: pd.DataFrame):
    severity_map = {"Info": 1, "Good": 1.4, "Warning": 2, "Critical": 3}
    chart_df = event_log.copy()
    chart_df["severity"] = chart_df["alert_level"].map(severity_map).fillna(1)
    return px.scatter(
        chart_df,
        x="timestamp",
        y="severity",
        size="number_people",
        color="alert_level",
        hover_data=["alert_message", "nearest_rescue_object", "estimated_closeness"],
        labels={"timestamp": "Timestamp (seconds)", "severity": "Alert level"},
        color_discrete_map={
            "Info": "#8aa4b8",
            "Good": "#087f5b",
            "Warning": "#bc7a00",
            "Critical": "#c72534",
        },
    )


def merge_alert_intervals(event_log: pd.DataFrame) -> pd.DataFrame:
    if event_log.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "alert_level", "alert_message", "frames"])

    intervals = []
    start_row = event_log.iloc[0]
    previous = start_row
    frame_count = 1

    for _, row in event_log.iloc[1:].iterrows():
        same_alert = (
            row["alert_level"] == previous["alert_level"]
            and row["alert_message"] == previous["alert_message"]
        )
        if same_alert:
            previous = row
            frame_count += 1
            continue

        intervals.append(
            {
                "start_time": start_row["timestamp"],
                "end_time": previous["timestamp"],
                "alert_level": start_row["alert_level"],
                "alert_message": start_row["alert_message"],
                "frames": frame_count,
            }
        )
        start_row = row
        previous = row
        frame_count = 1

    intervals.append(
        {
            "start_time": start_row["timestamp"],
            "end_time": previous["timestamp"],
            "alert_level": start_row["alert_level"],
            "alert_message": start_row["alert_message"],
            "frames": frame_count,
        }
    )
    return pd.DataFrame(intervals)


def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.title("🌊 Controls")
    st.sidebar.caption("Tune the analysis for the uploaded scene.")
    mode = st.sidebar.radio(
        "Analysis type",
        ["Water Image Analysis", "Water Video Analysis"],
        index=0,
    )
    person_confidence = st.sidebar.slider("Detection confidence threshold", 0.10, 0.90, 0.35, 0.05)
    object_confidence = st.sidebar.slider("Object confidence threshold", 0.10, 0.90, 0.25, 0.05)
    sampling_interval = st.sidebar.slider("Frame sampling interval", 1, 5, 2, 1)
    max_frames = st.sidebar.slider("Maximum frames to analyze", 1, 30, 10, 1)
    show_depth_map = st.sidebar.checkbox("Show depth map", value=True)
    show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
    st.sidebar.caption("Object search uses a fixed sea and rescue object list.")
    return {
        "mode": mode,
        "person_confidence": person_confidence,
        "object_confidence": object_confidence,
        "sampling_interval": sampling_interval,
        "max_frames": max_frames,
        "show_depth_map": show_depth_map,
        "show_boxes": show_boxes,
    }


def upload_section(mode: str) -> Optional[Any]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload Area")
    if mode == "Water Image Analysis":
        uploaded = st.file_uploader("Upload image: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.file_uploader("Upload video: MP4, MOV, AVI", type=["mp4", "mov", "avi"])
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded


def prepare_frames(uploaded_file: Any, settings: Dict[str, Any]) -> Tuple[bytes, List[Dict[str, Any]]]:
    file_bytes = uploaded_file.getvalue()
    if settings["mode"] == "Water Image Analysis":
        frames = [{"frame_index": 0, "timestamp": 0.0, "image": read_uploaded_image(file_bytes)}]
    else:
        frames = extract_video_frames(
            file_bytes,
            settings["sampling_interval"],
            settings["max_frames"],
        )
    return file_bytes, frames


def display_preview(uploaded_file: Any, frames: List[Dict[str, Any]], mode: str) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Scene Preview")
    if mode == "Water Image Analysis":
        st.image(frames[0]["image"], use_container_width=True)
    else:
        st.video(uploaded_file)
        st.caption("Sample frame used for visual checks")
        st.image(frames[0]["image"], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_person_detection(analyzed_frames: List[Dict[str, Any]], settings: Dict[str, Any]) -> pd.DataFrame:
    event_log = build_initial_event_log(analyzed_frames)
    total_people = int(event_log["number_people"].sum())
    any_person = total_people > 0
    top_confidence = max([average_confidence(frame["people"]) for frame in analyzed_frames] or [0])
    critical_frame = choose_critical_frame(analyzed_frames)
    selected = analyzed_frames[critical_frame]
    annotated = draw_detections(
        selected["image"],
        people=selected["people"],
        objects=None,
        show_boxes=settings["show_boxes"],
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("👤 Person Detection")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        render_metric_card("Person detected", "Yes" if any_person else "No")
    with col_b:
        render_metric_card("People detected", total_people)
    with col_c:
        render_metric_card("Confidence score", f"{top_confidence:.2f}" if top_confidence else "0.00")

    if any_person:
        render_badge("Person detected in water scene", "warn")
        if event_log["number_people"].max() > 1:
            render_badge("⚠️ Multiple persons detected", "danger")
    else:
        render_badge("No person detected in water scene.", "info")

    st.image(annotated, caption="Person detection result", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return event_log


def display_water_risk(event_log: pd.DataFrame) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚠️ Water Risk")
    total_people = int(event_log["number_people"].sum())
    max_people = int(event_log["number_people"].max()) if not event_log.empty else 0
    if total_people == 0:
        render_badge("No person detected in water scene.", "info")
        st.write("Continue monitoring or upload a clearer scene if risk is still suspected.")
    elif max_people > 1:
        render_badge("⚠️ Multiple people detected", "danger")
        st.write("Potential rescue attention required.")
    else:
        render_badge("⚠️ Potential rescue attention required.", "warn")
        st.write("A person is visible in the uploaded water scene.")
    st.markdown("</div>", unsafe_allow_html=True)


def display_rescue_search(
    analyzed_frames: List[Dict[str, Any]],
    event_log: pd.DataFrame,
    settings: Dict[str, Any],
    rescue_key: str,
) -> pd.DataFrame:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🛟 Find Sea Objects and Distance")
    prompt_labels = sea_object_labels()
    people_present = int(event_log["number_people"].sum()) > 0

    st.caption("Grounding DINO runs only when this button is pressed and searches common sea and rescue objects.")

    clicked = st.button(
        "Find Objects and Distance to Person",
        type="primary",
        disabled=not people_present,
        use_container_width=True,
    )

    if clicked:
        with st.spinner("Searching for sea objects and estimating relative depth..."):
            try:
                st.session_state["rescue_result"] = run_rescue_analysis(
                    analyzed_frames,
                    event_log,
                    prompt_labels,
                    settings["object_confidence"],
                )
                st.session_state["rescue_key"] = rescue_key
            except Exception as exc:
                st.session_state.pop("rescue_result", None)
                st.session_state.pop("rescue_key", None)
                st.error("Object-distance analysis could not finish. Check the model dependencies and try again.")
                st.caption(str(exc))

    if not people_present:
        st.info("Run object-distance search after a person is detected in the water scene.")

    result = st.session_state.get("rescue_result")
    if result and st.session_state.get("rescue_key") == rescue_key:
        alert_style = {
            "Critical": "danger",
            "Warning": "warn",
            "Good": "good",
            "Info": "info",
        }.get(result["alert_level"], "info")
        render_badge(result["alert_message"], alert_style)
        st.caption(f"Analyzed frame timestamp: {result['selected_timestamp']} seconds")

        col_a, col_b, col_c = st.columns(3)
        nearest = result["nearest"]
        with col_a:
            render_metric_card("Nearest object", nearest["object"].label if nearest else "Not found")
        with col_b:
            render_metric_card("Approx. closeness score", nearest["closeness"] if nearest else "N/A")
        with col_c:
            render_metric_card("Confidence level", nearest["confidence_level"] if nearest else "N/A")

        st.image(result["annotated"], caption="Sea object detection result", use_container_width=True)
        distance_rows = []
        for item in result.get("object_distances", []):
            distance_rows.append(
                {
                    "object": item["object"].label,
                    "object_confidence": round(item["object"].confidence, 3),
                    "relative_closeness": item["closeness"],
                    "distance_score": item["distance_score"],
                    "confidence_level": item["confidence_level"],
                }
            )
        if distance_rows:
            st.caption("Detected object distance from person")
            st.dataframe(pd.DataFrame(distance_rows), use_container_width=True, hide_index=True)
        if settings["show_depth_map"]:
            st.image(result["depth_preview"], caption="ZoeDepth near/far preview", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
        return result["event_log"]

    st.markdown("</div>", unsafe_allow_html=True)
    return event_log


def display_timeline_and_log(event_log: pd.DataFrame, mode: str) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    title = "Timeline and Event Log" if mode == "Water Video Analysis" else "Event Log"
    st.subheader(title)
    display_columns = [
        "timestamp",
        "person_detected",
        "number_people",
        "rescue_object_status",
        "nearest_rescue_object",
        "estimated_closeness",
        "alert_level",
        "alert_message",
    ]
    st.dataframe(event_log[display_columns], use_container_width=True, hide_index=True)
    if mode == "Water Video Analysis":
        st.caption("Merged alert intervals")
        st.dataframe(merge_alert_intervals(event_log), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_summary(event_log: pd.DataFrame) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📍 Summary")
    summary = summarize_events(event_log)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total frames analyzed", summary["total_frames_analyzed"])
    with col2:
        render_metric_card("Total person detections", summary["total_person_detections"])
    with col3:
        render_metric_card("Object-distance searches", summary["total_rescue_object_searches"])

    col4, col5, col6 = st.columns(3)
    with col4:
        render_metric_card("Nearest object found", summary["nearest_object_found"])
    with col5:
        render_metric_card("Critical alert count", summary["critical_alert_count"])
    with col6:
        render_metric_card("Most recent alert", summary["most_recent_alert"])
    st.markdown("</div>", unsafe_allow_html=True)


def display_charts(event_log: pd.DataFrame, mode: str) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Charts and Visuals")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(style_chart(create_person_count_chart(event_log)), use_container_width=True)
    with col2:
        st.plotly_chart(style_chart(create_rescue_count_chart(event_log)), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(style_chart(create_alert_chart(event_log)), use_container_width=True)
    with col4:
        if mode == "Water Video Analysis":
            st.plotly_chart(style_chart(create_timeline_chart(event_log)), use_container_width=True)
        else:
            st.info("Timeline chart appears for video analysis.")
    st.markdown("</div>", unsafe_allow_html=True)


def display_exports(event_log: pd.DataFrame) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Export")
    summary_report = pd.DataFrame([summarize_events(event_log)])
    critical_alerts = event_log[event_log["alert_level"] == "Critical"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download event log CSV",
            data=event_log.to_csv(index=False).encode("utf-8"),
            file_name="water_rescue_event_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download summary report CSV",
            data=summary_report.to_csv(index=False).encode("utf-8"),
            file_name="water_rescue_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "Download critical alerts CSV",
            data=critical_alerts.to_csv(index=False).encode("utf-8"),
            file_name="water_rescue_critical_alerts.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def reset_rescue_result_if_needed(rescue_key: str) -> None:
    if st.session_state.get("rescue_key") != rescue_key:
        st.session_state.pop("rescue_result", None)


def main() -> None:
    set_page_config()
    inject_css()
    render_hero()
    settings = sidebar_controls()
    uploaded_file = upload_section(settings["mode"])

    if uploaded_file is None:
        st.info("Upload an image or video to begin water-scene analysis.")
        st.markdown(
            '<div class="footer">Faster visual awareness can support faster rescue response.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        file_bytes, frames = prepare_frames(uploaded_file, settings)
    except Exception as exc:
        st.error("This file could not be analyzed. Please upload a valid water-scene image or video.")
        st.caption(str(exc))
        return

    person_key = file_signature(
        file_bytes,
        settings["mode"],
        settings["person_confidence"],
        settings["sampling_interval"],
        settings["max_frames"],
    )
    rescue_key = file_signature(
        file_bytes,
        settings["mode"],
        settings["person_confidence"],
        settings["object_confidence"],
        settings["sampling_interval"],
        settings["max_frames"],
        ",".join(sea_object_labels()),
    )
    reset_rescue_result_if_needed(rescue_key)

    display_preview(uploaded_file, frames, settings["mode"])

    try:
        if st.session_state.get("person_key") != person_key:
            with st.spinner("Detecting people in the water scene..."):
                st.session_state["analyzed_frames"] = analyze_frames_for_people(
                    frames,
                    settings["person_confidence"],
                )
                st.session_state["person_key"] = person_key
        analyzed_frames = st.session_state["analyzed_frames"]
    except Exception as exc:
        st.error("Person detection could not finish. Check the YOLO11 dependency and try again.")
        st.caption(str(exc))
        return

    event_log = display_person_detection(analyzed_frames, settings)
    display_water_risk(event_log)
    event_log = display_rescue_search(analyzed_frames, event_log, settings, rescue_key)
    display_timeline_and_log(event_log, settings["mode"])
    display_summary(event_log)
    display_charts(event_log, settings["mode"])
    display_exports(event_log)
    st.markdown(
        '<div class="footer">Faster visual awareness can support faster rescue response.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
