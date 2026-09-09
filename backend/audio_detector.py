"""Lazy adapter around the bundled Wav2Vec2 audio deepfake pipeline."""

import importlib.util
import sys
import tempfile
from pathlib import Path


PIPELINE_PATH = Path(__file__).with_name("audio_deepfake_pipeline (1).py")
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        spec = importlib.util.spec_from_file_location("audio_deepfake_pipeline", PIPELINE_PATH)
        if not spec or not spec.loader:
            raise RuntimeError("The bundled audio deepfake pipeline could not be loaded.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _pipeline = module
    return _pipeline


def analyze_audio_pipeline(audio_path: str, uploads_dir: str) -> dict:
    """Run the project audio model and normalize its output for the API."""
    try:
        pipeline = _load_pipeline()
    except ModuleNotFoundError as exc:
        return {"error": f"Audio model dependency is missing: {exc.name}. Install backend/requirements.txt."}
    except Exception as exc:
        return {"error": f"Unable to initialize the audio model: {exc}"}

    output_dir = tempfile.mkdtemp(prefix="audio_", dir=uploads_dir)
    try:
        result = pipeline.detect_audio_deepfake(audio_path, output_dir=output_dir)
        synthetic_probability = float(result["probabilities"]["synthetic"])
        return {
            "verdict": "AI-Generated / Manipulated" if result["label"] == "Synthetic" else "Likely Authentic",
            "visual_confidence_score": round(synthetic_probability * 100, 2),
            "peak_anomaly_score": round(synthetic_probability * 100, 2),
            "frames_analyzed": 1,
            "audio_extracted": True,
            "audio_path": None,
            "evidence_frame_base64": None,
            "duration_seconds": result.get("duration_seconds"),
            "detected_signals": result.get("detected_signals", []),
            "model_id": result.get("model_id"),
            "artifact_dir": output_dir,
        }
    except Exception as exc:
        return {"error": f"Audio analysis failed: {exc}", "artifact_dir": output_dir}
