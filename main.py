"""Root entrypoint for single-deploy and local execution of TrueSight AI.

Runs the FastAPI backend and serves all static frontend pages from one origin.
"""

import os
from pathlib import Path
import uvicorn
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.index import app

ROOT_DIR = Path(__file__).resolve().parent


# Route friendly URL paths
@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/analyze", include_in_schema=False)
@app.get("/analyze.html", include_in_schema=False)
def analyze_page():
    return FileResponse(ROOT_DIR / "analyze.html")


@app.get("/reports", include_in_schema=False)
@app.get("/reports.html", include_in_schema=False)
def reports_page():
    return FileResponse(ROOT_DIR / "reports.html")


@app.get("/history", include_in_schema=False)
@app.get("/history.html", include_in_schema=False)
def history_page():
    return FileResponse(ROOT_DIR / "history.html")


@app.get("/how-it-works", include_in_schema=False)
@app.get("/how-it-works.html", include_in_schema=False)
def how_it_works_page():
    return FileResponse(ROOT_DIR / "how-it-works.html")


# Serve static assets (CSS, JS, images) from project root
app.mount("/", StaticFiles(directory=ROOT_DIR, html=True), name="static-root")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting TrueSight AI on http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
