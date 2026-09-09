import os
import shutil
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import your machine learning pipeline
from detect import analyze_video_pipeline

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


def cleanup_files(*file_paths):
    """Deletes temporary files from the server after processing."""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Cleaned up: {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")


# 3. Health Check Route
@app.get("/")
def read_root():
    return {"status": "online", "message": "Deepfake Detection API is running."}


# 4. The Main Deepfake Analysis Route
@app.post("/api/analyze-deepfake")
async def analyze_deepfake_video(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    # Reject non-video files early
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a video."
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
        # RUN THE ML MODEL
        # We use asyncio.to_thread because OpenCV and PyTorch are synchronous and blocking.
        # This prevents the AI model from freezing the whole web server for other users.
        results = await asyncio.to_thread(analyze_video_pipeline, temp_video_path, 1)

        # Schedule cleanup of the video and the audio file AFTER the response is sent
        audio_path = results.get("audio_path")
        background_tasks.add_task(cleanup_files, temp_video_path, audio_path)

        # Check if the AI script returned an error internally
        if "error" in results:
            return JSONResponse(status_code=400, content=results)

        # Return success data to the frontend
        return results

    except Exception as e:
        background_tasks.add_task(cleanup_files, temp_video_path)
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Starts the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
