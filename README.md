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
