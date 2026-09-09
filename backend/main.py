import os
import shutil
import tempfile
import asyncio
import html
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse
import httpx
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import your machine learning pipeline
from audio_detector import analyze_audio_pipeline
from detect import analyze_image_pipeline, analyze_video_pipeline

# 1. Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Backend for analyzing videos for AI manipulation.",
    version="1.0.0",
)

# 2. Configure CORS (CRITICAL FOR FRONTEND)
# This allows your frontend (e.g., React, Vue, or plain HTML) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # In production, change "*" to your frontend URL (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
SOCIAL_HOSTS = ("instagram.com", "facebook.com", "x.com", "twitter.com", "tiktok.com", "youtube.com", "youtu.be")
MAX_SOCIAL_RESPONSE_BYTES = 10 * 1024 * 1024


class SocialLinkRequest(BaseModel):
    url: str


def cleanup_files(*file_paths):
    """Deletes temporary files from the server after processing."""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"Cleaned up: {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")


def is_public_host(hostname: str) -> bool:
    """Reject local/private hosts before downloading media from a user-supplied URL."""
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (socket.gaierror, ValueError):
        return False


def fetch_public_url(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch a public URL with bounded redirects and response size."""
    current_url = url
    for _ in range(4):
        parsed = urlparse(current_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not is_public_host(parsed.hostname):
            raise HTTPException(status_code=400, detail="The link must resolve to a public host.")
        response = client.get(current_url, follow_redirects=False)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise HTTPException(status_code=400, detail="The social link returned an invalid redirect.")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        if int(response.headers.get("content-length", 0) or 0) > MAX_SOCIAL_RESPONSE_BYTES:
            raise HTTPException(status_code=413, detail="The social preview is too large to process.")
        if len(response.content) > MAX_SOCIAL_RESPONSE_BYTES:
            raise HTTPException(status_code=413, detail="The social preview is too large to process.")
        return response
    raise HTTPException(status_code=400, detail="The social link redirected too many times.")


def extract_open_graph_image(post_url: str) -> tuple[str, bytes]:
    parsed = urlparse(post_url)
    hostname = (parsed.hostname or "").lower()
    if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in SOCIAL_HOSTS):
        raise HTTPException(status_code=400, detail="Only public Instagram, Facebook, X, TikTok, and YouTube links are supported.")

    headers = {"User-Agent": "TrueSightAI/1.0 (+media-preview)", "Accept": "text/html,application/xhtml+xml"}
    with httpx.Client(headers=headers, timeout=15.0) as client:
        page = fetch_public_url(client, post_url)
        if "html" not in page.headers.get("content-type", "").lower():
            raise HTTPException(status_code=400, detail="The social link did not return a public web page.")
        match = re.search(r'<meta[^>]+(?:property|name)=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)', page.text, re.IGNORECASE)
        if not match:
            match = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:image(?::secure_url)?["\']', page.text, re.IGNORECASE)
        if not match:
            raise HTTPException(status_code=422, detail="No public preview image was found for this social link.")
        image_url = html.unescape(urljoin(str(page.url), match.group(1)))
        image = fetch_public_url(client, image_url)
        if not image.headers.get("content-type", "").lower().startswith("image/"):
            raise HTTPException(status_code=422, detail="The social preview is not an image.")
        return image_url, image.content


# 3. Health Check Route
@app.get("/api/health")
def health_check():
    return {"status": "online", "message": "Deepfake Detection API is running."}


# 4. The Main Deepfake Analysis Route
@app.post("/api/analyze-deepfake")
async def analyze_deepfake_video(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith(("video/", "audio/", "image/")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Upload an image, video, or audio file."
        )

    # Create a safe temporary file in the uploads directory
    temp_video_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")

    try:
        # Save the uploaded file in chunks (prevents server memory crashes on large files)
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        cleanup_files(temp_video_path)
        raise HTTPException(status_code=500, detail="Failed to save uploaded video.")
    finally:
        file.file.close()

    try:
        if file.content_type.startswith("video/"):
            results = await asyncio.to_thread(analyze_video_pipeline, temp_video_path, 1)
        elif file.content_type.startswith("audio/"):
            results = await asyncio.to_thread(analyze_audio_pipeline, temp_video_path, UPLOAD_DIR)
        else:
            results = await asyncio.to_thread(analyze_image_pipeline, temp_video_path)

        # Schedule cleanup of the video and the audio file AFTER the response is sent
        background_tasks.add_task(cleanup_files, temp_video_path, results.get("audio_path"), results.get("artifact_dir"))

        # Check if the AI script returned an error internally
        if "error" in results:
            return JSONResponse(status_code=400, content=results)

        # Return success data to the frontend
        return results

    except Exception as e:
        background_tasks.add_task(cleanup_files, temp_video_path)
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")


@app.post("/api/analyze-social-image")
async def analyze_social_image(request: SocialLinkRequest, background_tasks: BackgroundTasks):
    """Extract and analyze the public Open Graph preview image of a social-media post."""
    image_url, image_bytes = await asyncio.to_thread(extract_open_graph_image, request.url)
    suffix = Path(urlparse(image_url).path).suffix or ".jpg"
    temp_image = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False)
    try:
        temp_image.write(image_bytes)
        temp_image.close()
        results = await asyncio.to_thread(analyze_image_pipeline, temp_image.name)
        if "error" in results:
            raise HTTPException(status_code=422, detail=results["error"])
        results["source_url"] = request.url
        results["extracted_image_url"] = image_url
        background_tasks.add_task(cleanup_files, temp_image.name)
        return results
    except HTTPException:
        background_tasks.add_task(cleanup_files, temp_image.name)
        raise
    except Exception as exc:
        background_tasks.add_task(cleanup_files, temp_image.name)
        raise HTTPException(status_code=500, detail=f"Social image analysis failed: {exc}")


@app.get("/analyze.html", include_in_schema=False)
@app.get("/analyse.html", include_in_schema=False)
def analyze_page_redirect():
    return RedirectResponse("/analyse-page.html")


@app.get("/reports.html", include_in_schema=False)
def reports_page_redirect():
    return RedirectResponse("/report.html")


@app.get("/how-it-works.html", include_in_schema=False)
def how_it_works_page_redirect():
    return RedirectResponse("/howitworkd.html")


# The frontend and API share one origin when served through Uvicorn.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # Starts the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
