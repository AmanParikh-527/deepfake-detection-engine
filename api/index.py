"""Dual-Model TrueSight AI Engine: Global Generative Detector + SigLIP 2 Face Deepfake Detector.

Combines:
1. umm-maybe/AI-image-detector (Global diffusion, Midjourney, DALL-E, SD, Flux detection)
2. prithivMLmods/Deepfake-Detect-Siglip2 (Facial manipulation and deepfake detection on cropped faces)
"""

import base64
import html
import ipaddress
import os
import re
import socket
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import cv2
import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, HttpUrl
from transformers import pipeline

app = FastAPI(title="TrueSight AI — Dual Model Deepfake Engine", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_REMOTE_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")


class SocialLinkRequest(BaseModel):
    url: HttpUrl


@lru_cache(maxsize=1)
def get_global_ai_detector():
    """umm-maybe/AI-image-detector: evaluates overall image for generative/diffusion artifacts."""
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector",
        device=device,
    )


@lru_cache(maxsize=1)
def get_facial_deepfake_detector():
    """prithivMLmods/Deepfake-Detect-Siglip2: evaluates facial landmarks and face-swap manipulation."""
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "image-classification",
        model="prithivMLmods/Deepfake-Detect-Siglip2",
        device=device,
    )


@lru_cache(maxsize=1)
def get_face_cascade():
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        return None
    return cascade


def is_public_host(hostname: str) -> bool:
    """Block local/private addresses before requesting a user-supplied URL."""
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        return bool(addresses) and all(
            ipaddress.ip_address(address[4][0]).is_global for address in addresses
        )
    except (socket.gaierror, ValueError):
        return False


def fetch_public_url(client: httpx.Client, url: str) -> httpx.Response:
    current_url = url
    for _ in range(4):
        parsed = urlparse(current_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or not is_public_host(parsed.hostname)
        ):
            raise HTTPException(400, "The link must resolve to a public host.")
        response = client.get(current_url, follow_redirects=False)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise HTTPException(400, "The link returned an invalid redirect.")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        if int(response.headers.get("content-length", 0) or 0) > MAX_REMOTE_IMAGE_BYTES:
            raise HTTPException(413, "The linked image is too large (max 4 MB).")
        if len(response.content) > MAX_REMOTE_IMAGE_BYTES:
            raise HTTPException(413, "The linked image is too large (max 4 MB).")
        return response
    raise HTTPException(400, "The link redirected too many times.")


def extract_open_graph_image(post_url: str) -> tuple[str, bytes]:
    parsed = urlparse(post_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not is_public_host(parsed.hostname)
    ):
        raise HTTPException(400, "The link must resolve to a public host.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 TrueSightAI/2.5"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
    }
    with httpx.Client(headers=headers, timeout=15.0) as client:
        initial_resp = fetch_public_url(client, post_url)
        content_type = initial_resp.headers.get("content-type", "").lower()

        # 1. Direct image link
        if content_type.startswith("image/"):
            return str(initial_resp.url), initial_resp.content

        # 2. Web page / Social post preview extraction
        if "html" in content_type:
            patterns = (
                r'<meta[^>]+(?:property|name)=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:image(?::secure_url)?["\']',
                r'<meta[^>]+(?:property|name)=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']twitter:image(?::src)?["\']',
                r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)',
            )
            match = next(
                (
                    re.search(pat, initial_resp.text, re.IGNORECASE)
                    for pat in patterns
                    if re.search(pat, initial_resp.text, re.IGNORECASE)
                ),
                None,
            )

            # Fallback for YouTube links if og:image isn't extracted from static HTML
            netloc = parsed.netloc.lower()
            if not match and ("youtube.com" in netloc or "youtu.be" in netloc):
                video_id = None
                if "youtu.be" in netloc:
                    video_id = parsed.path.strip("/").split("/")[0]
                elif "youtube.com" in netloc:
                    query_params = dict(
                        q.split("=") for q in parsed.query.split("&") if "=" in q
                    )
                    video_id = query_params.get("v")
                if video_id:
                    thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                    try:
                        thumb_resp = fetch_public_url(client, thumb_url)
                        if thumb_resp.status_code == 200 and thumb_resp.headers.get(
                            "content-type", ""
                        ).startswith("image/"):
                            return thumb_url, thumb_resp.content
                    except Exception:
                        pass

            if not match:
                raise HTTPException(
                    422, "No public preview image was found for this link."
                )

            image_url = html.unescape(urljoin(str(initial_resp.url), match.group(1)))
            image_resp = fetch_public_url(client, image_url)
            if (
                not image_resp.headers.get("content-type", "")
                .lower()
                .startswith("image/")
            ):
                raise HTTPException(422, "The extracted preview is not a valid image.")
            return image_url, image_resp.content

        raise HTTPException(
            400,
            "The link must point to a direct image or a public web page with preview images.",
        )


def extract_face_crop(image: Image.Image) -> tuple[Image.Image, bool]:
    """Detect primary face in image and crop with margin for face-level evaluation."""
    try:
        cascade = get_face_cascade()
        if cascade is None:
            return image, False

        img_np = np.array(image)
        if len(img_np.shape) == 2:
            gray = img_np
        elif img_np.shape[2] == 4:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return image, False

        # Select largest face by area
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]

        img_h, img_w = img_np.shape[:2]
        x_margin, y_margin = int(w * 0.15), int(h * 0.15)
        x1 = max(0, x - x_margin)
        y1 = max(0, y - y_margin)
        x2 = min(img_w, x + w + x_margin)
        y2 = min(img_h, y + h + y_margin)

        cropped = image.crop((x1, y1, x2, y2))
        return cropped, True
    except Exception:
        return image, False


def analyze_image(image_bytes: bytes) -> dict:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image dimensions are too large to analyze.")
            image = source.convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(400, "Upload a valid image file.") from error

    # 1. Model A: Global Generative AI Detector (umm-maybe/AI-image-detector)
    # Detects global Midjourney, DALL-E, Stable Diffusion, and Flux generation patterns
    global_detector = get_global_ai_detector()
    global_results = global_detector(image)
    global_ai_score = next(
        (
            item["score"]
            for item in global_results
            if item["label"].lower() in ("artificial", "fake")
        ),
        0.0,
    )

    # 2. Model B: Facial Deepfake Detector (prithivMLmods/Deepfake-Detect-Siglip2)
    # Detects face swaps, GAN blending boundaries, and facial landmark anomalies
    crop_for_eval, face_detected = extract_face_crop(image)
    face_detector = get_facial_deepfake_detector()
    face_results = face_detector(crop_for_eval)
    face_fake_score = next(
        (
            item["score"]
            for item in face_results
            if item["label"].lower() in ("fake", "artificial", "deepfake")
        ),
        0.0,
    )

    # 3. Ensemble Synthesis
    # If a face is present, combine global generator score and facial manipulation score.
    # Otherwise, rely on global generator score.
    if face_detected:
        final_fake_score = max(global_ai_score, face_fake_score)
    else:
        final_fake_score = global_ai_score

    evidence = image.copy()
    evidence.thumbnail((1200, 1200))
    evidence_buffer = BytesIO()
    evidence.save(evidence_buffer, format="JPEG", quality=80, optimize=True)

    is_manipulated = final_fake_score > 0.50

    return {
        "verdict": (
            "AI-Generated / Manipulated" if is_manipulated else "Likely Authentic"
        ),
        "visual_confidence_score": round(final_fake_score * 100, 2),
        "peak_anomaly_score": round(max(global_ai_score, face_fake_score) * 100, 2),
        "global_ai_score": round(global_ai_score * 100, 2),
        "facial_deepfake_score": (
            round(face_fake_score * 100, 2) if face_detected else None
        ),
        "face_detected": face_detected,
        "models_used": 2 if face_detected else 1,
        "model_ensemble": [
            "umm-maybe/AI-image-detector (Global AI)",
            "prithivMLmods/Deepfake-Detect-Siglip2 (Face Deepfake)",
        ],
        "frames_analyzed": 1,
        "evidence_frame_base64": base64.b64encode(evidence_buffer.getvalue()).decode(
            "ascii"
        ),
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "mode": "dual-model-ensemble",
        "models": [
            "umm-maybe/AI-image-detector",
            "prithivMLmods/Deepfake-Detect-Siglip2",
        ],
    }


@app.post("/api/analyze-deepfake")
async def analyze_uploaded_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image uploads are supported.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Images must be 4 MB or smaller.")
    return analyze_image(content)


@app.post("/api/analyze-social-image")
def analyze_social_image(request: SocialLinkRequest):
    image_url, image_bytes = extract_open_graph_image(str(request.url))
    result = analyze_image(image_bytes)
    result["source_url"] = str(request.url)
    result["extracted_image_url"] = image_url
    return result
