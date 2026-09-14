# TrueSight AI — Deepfake Image Detection Engine

A consolidated, single-deploy deepfake image detector built with FastAPI, PyTorch, Hugging Face Transformers, OpenCV, and a modern responsive frontend.

Features:

- **Pure Image Detection**: High-accuracy deepfake and AI manipulation detection (using Vision Transformer models).
- **Intelligent Face Extraction**: Automatic facial detection and cropping (via Haar Cascades) to evaluate facial artifacts with high precision, falling back to full-image analysis when no faces are detected.
- **Link & Social Preview Extraction**: Inspect images directly from public URLs, including direct image links and social posts (YouTube thumbnails, Instagram, X/Twitter, TikTok, Facebook, Reddit).
- **Interactive Forensic Reports**: Attention heatmap toggles, confidence score breakdown, signal analysis, and printable PDF export.
- **Local Examination History**: Browse recent checks stored safely in browser storage and reload them into the report view.
- **Single-Deploy Architecture**: Unified backend and frontend that runs locally with a single command (`python main.py`), in containers, or directly on Vercel Serverless.

---

## 🚀 Deployment Guide: GitHub & Vercel

### Step 1: Commit and Push to GitHub

From your terminal in the project root:

```bash
# Check git status
git status

# Stage all updated files and deleted legacy files
git add -A

# Commit your changes
git commit -m "Consolidate to single-deploy pure image detection engine"

# Push to your GitHub repository
git push origin main
```

_(If you are pushing to a new repository for the first time, run `git remote add origin https://github.com/<your-username>/<your-repo>.git` and `git push -u origin main`)_

---

### Step 2: Deploy to Vercel

1. **Sign in to Vercel**:
   Go to [vercel.com](https://vercel.com) and log in with your GitHub account.

2. **Add New Project**:
   - Click **"Add New..."** -> **"Project"**.
   - Under **"Import Git Repository"**, select your `deepfake-detection-engine` repository.

3. **Configure the Project**:
   - **Framework Preset**: Select **"Other"** (Vercel automatically detects the static HTML files and Python serverless function in `api/index.py`).
   - **Root Directory**: Leave as `./` (the repository root).
   - **Build Command**: Leave empty (no build step is needed).
   - **Output Directory**: Leave empty.
   - **Environment Variables**: None required by default. _(Optional: You can add `HF_TOKEN` if you want higher rate limits when downloading Hugging Face models)_.

4. **Adjust Function Settings**:
   - In your Vercel Project Dashboard under **Settings → Functions**:
     - Ensure **Fluid Compute** is active.
     - `vercel.json` automatically sets `maxDuration` to `300` seconds for `api/index.py` so the model has ample time to download on cold starts.

5. **Click "Deploy"**:
   - Vercel will deploy your project.
   - Once complete, you will receive a production URL (e.g. `https://your-project.vercel.app`).
   - Open your deployed URL — your full interface is live at `/` and the backend is live at `/api/health`.

---

## 💻 Running Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Unified Server

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload --port 8000
```

### 3. Open in Browser

Open [http://localhost:8000](http://localhost:8000) to use the full application.

---

## 📁 Project Structure

```
├── api/
│   ├── index.py                    # Serverless FastAPI application & ML pipelines
│   ├── haarcascade_frontalface_default.xml # OpenCV face detection cascade
│   └── __init__.py
├── main.py                         # Unified local and standalone server entrypoint
├── index.html                      # Landing & About page
├── analyze.html                    # Analysis intake (file upload & link import)
├── analyze.js                      # Intake handler & progress engine
├── reports.html                    # Forensic verdict, attention heatmap & PDF export
├── reports.js                      # Report renderer & client-side PDF generation
├── history.html                    # Saved examinations table
├── history-page.js                 # History table renderer & navigation
├── how-it-works.html               # Technical architecture & pipeline explainer
├── styles.css                      # Unified design system
├── api-config.js                   # Optional API URL configuration
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment configuration
└── README.md                       # Project documentation
```

---

## 🛡️ Security & Privacy

- **Zero Data Retention**: Uploaded media is analyzed in memory or temporary files and discarded immediately after processing.
- **SSRF Protection**: All user-supplied URLs are strictly validated against private/internal IP ranges before fetching.
