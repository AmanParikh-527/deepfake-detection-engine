import argparse
import base64
import os
import subprocess
import sys
import tempfile
import cv2
import torch
from PIL import Image
from transformers import pipeline

# 1. Device Acceleration & Model Setup
device = (
    0
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else -1)
)
detector = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
    device=device,
)

# 2. Bulletproof OpenCV Cascade Loading
# This guarantees it looks in the exact folder where this detect.py script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("CRITICAL ERROR: Could not load the Haar Cascade XML file.")
    print(f"I am looking for it exactly here:\n{cascade_path}")
    print("\nPlease make sure 'haarcascade_frontalface_default.xml' is in that folder!")
    sys.exit(1)


def extract_audio(video_path: str, output_audio_path: str = None) -> str | None:
    """Strips the audio track from the video for acoustic anomaly checks."""
    if output_audio_path is None:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_audio_path = temp_audio.name
        temp_audio.close()

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
        "-y",
    ]
    try:
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return output_audio_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_cropped_face(rgb_frame):
    """Detects and extracts the largest face with a margin."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = faces[0]

    frame_h, frame_w, _ = rgb_frame.shape
    x_margin, y_margin = int(w * 0.1), int(h * 0.1)

    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(frame_w, x + w + x_margin)
    y2 = min(frame_h, y + h + y_margin)

    return rgb_frame[y1:y2, x1:x2]


def encode_frame_to_base64(bgr_frame) -> str:
    """Converts a standard BGR OpenCV frame to a base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", bgr_frame)
    return base64.b64encode(buffer).decode("utf-8")


def analyze_video_pipeline(video_path: str, sample_rate_fps: int = 1) -> dict:
    """Processes video frames and evaluates face authenticity."""
    if not os.path.exists(video_path):
        return {"error": f"File not found: '{video_path}'"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Invalid or unreadable video file format."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30
    frame_interval = max(1, fps // sample_rate_fps)

    frame_count = 0
    fake_scores = []
    peak_fake_score = 0.0
    peak_frame_b64 = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_face = get_cropped_face(rgb_frame)

            if cropped_face is not None and cropped_face.size > 0:
                pil_image = Image.fromarray(cropped_face)
                results = detector(pil_image)

                for res in results:
                    if res["label"].lower() == "fake":
                        score = res["score"]
                        fake_scores.append(score)

                        if score > peak_fake_score:
                            peak_fake_score = score
                            peak_frame_b64 = encode_frame_to_base64(frame)
                        break

        frame_count += 1

    cap.release()

    audio_path = extract_audio(video_path)

    if not fake_scores:
        return {"error": "No faces detected in the analyzed frames."}

    avg_fake_confidence = sum(fake_scores) / len(fake_scores)
    is_manipulated = avg_fake_confidence > 0.50

    return {
        "verdict": (
            "AI-Generated / Manipulated" if is_manipulated else "Likely Authentic"
        ),
        "visual_confidence_score": round(avg_fake_confidence * 100, 2),
        "peak_anomaly_score": round(peak_fake_score * 100, 2),
        "frames_analyzed": len(fake_scores),
        "audio_extracted": audio_path is not None,
        "audio_path": audio_path,
        "evidence_frame_base64": peak_frame_b64,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a video for deepfake manipulation."
    )
    parser.add_argument(
        "--video", type=str, default="sample.MOV", help="Path to input video file"
    )
    parser.add_argument(
        "--fps", type=int, default=1, help="Frame sampling rate per second"
    )
    args = parser.parse_args()

    print(f"Analyzing: {args.video} ...")
    analysis = analyze_video_pipeline(args.video, sample_rate_fps=args.fps)

    if "error" in analysis:
        print(f"Analysis failed: {analysis['error']}")
        sys.exit(1)

    print("\n--- Detection Results ---")
    print(f"Verdict:              {analysis['verdict']}")
    print(f"Confidence Score:     {analysis['visual_confidence_score']}%")
    print(f"Peak Anomaly Score:   {analysis['peak_anomaly_score']}%")
    print(f"Faces/Frames Checked: {analysis['frames_analyzed']}")
    print(f"Audio Extracted:      {analysis['audio_extracted']}")
    if analysis.get("audio_path"):
        print(f"Extracted Audio Path: {analysis['audio_path']}")
