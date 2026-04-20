from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import librosa
except ImportError:
    librosa = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import cv2
except ImportError:
    cv2 = None
from flask import Flask, jsonify, render_template, request

try:
    from fer.fer import FER
except (ImportError, ModuleNotFoundError):
    FER = None

from src.song_recommender import SongRecommender

ROOT = Path(__file__).resolve().parent
CATALOGUE_PATH = ROOT / "data" / "marathi_songs.json"
FINAL_EMOTIONS = {"happy", "sad", "angry", "neutral", "surprise"}
EMOTION_NORMALIZATION = {
    "disgust": "angry",
    "fear": "surprise",
    "sadness": "sad",
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "surprise": "surprise",
}

app = Flask(__name__)

face_detector = None
if FER is not None:
    try:
        face_detector = FER(mtcnn=False)
    except Exception:
        face_detector = None

recommender = SongRecommender(CATALOGUE_PATH)


def _strip_data_url(data_url: str) -> bytes:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url
    return base64.b64decode(payload)


def _normalize_emotion(label: str) -> str:
    normalized = label.strip().lower()
    return EMOTION_NORMALIZATION.get(normalized, normalized if normalized in FINAL_EMOTIONS else "neutral")


def _extract_youtube_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    # Accept plain IDs or known YouTube URL forms.
    url = url.strip()
    if len(url) == 11 and " " not in url:
        return url

    if "youtu.be/" in url:
        return url.split("youtu.be/", 1)[1].split("?")[0].split("/")[0]
    if "youtube.com/watch" in url and "v=" in url:
        query = url.split("?", 1)[1]
        for part in query.split("&"):
            if part.startswith("v="):
                return part.split("=", 1)[1]
    if "youtube.com/embed/" in url:
        return url.split("embed/", 1)[1].split("?")[0]
    return None


def detect_face_emotion(image_data_url: str) -> Optional[Dict[str, Any]]:
    if face_detector is None:
        return None
    try:
        image_bytes = _strip_data_url(image_data_url)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detector.detect_emotions(rgb_frame)
        if not detections:
            return None

        top_face = max(
            detections,
            key=lambda face: max(face["emotions"].values()) if face.get("emotions") else 0,
        )
        emotions = top_face.get("emotions", {})
        if not emotions:
            return None

        label, score = max(emotions.items(), key=lambda item: item[1])
        return {
            "label": _normalize_emotion(label),
            "score": float(score),
            "raw": emotions,
        }
    except Exception:
        return None


def detect_voice_emotion(audio_data_url: str) -> Optional[Dict[str, Any]]:
    try:
        audio_bytes = _strip_data_url(audio_data_url)
        with io.BytesIO(audio_bytes) as buffer:
            y, sr = librosa.load(buffer, sr=22050, mono=True)

        if y.size == 0:
            return None

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = float(np.mean(librosa.feature.rms(y=y, frame_length=2048, hop_length=512)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        if tempo > 130 and rms > 0.038 and centroid > 2300:
            label = "happy"
        elif tempo > 140 and rms > 0.05 and centroid > 2600:
            label = "angry"
        elif tempo < 90 and rms < 0.028:
            label = "sad"
        elif centroid > 3000 and zcr > 0.12:
            label = "surprise"
        else:
            label = "neutral"

        confidence = float(np.clip(0.33 + (rms * 2.5) + (tempo / 300.0) + (zcr * 0.4), 0.25, 0.95))
        return {
            "label": _normalize_emotion(label),
            "score": confidence,
            "features": {
                "tempo": round(float(tempo), 1),
                "rms": round(rms, 4),
                "centroid": round(centroid, 1),
                "zcr": round(zcr, 4),
            },
        }
    except Exception:
        return None


def combine_emotions(face_emotion: Optional[Dict[str, Any]], voice_emotion: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if face_emotion is None and voice_emotion is None:
        return {"label": "neutral", "score": 0.5, "source": "fallback"}
    if face_emotion is None:
        return {**voice_emotion, "source": "voice"}
    if voice_emotion is None:
        return {**face_emotion, "source": "face"}

    face_score = float(face_emotion.get("score", 0.0))
    voice_score = float(voice_emotion.get("score", 0.0))
    face_label = face_emotion["label"]
    voice_label = voice_emotion["label"]

    if face_label == voice_label:
        return {
            "label": face_label,
            "score": float(np.clip(face_score + 0.18, 0.0, 0.99)),
            "source": "combined",
        }

    if face_score >= 0.72 and face_score >= voice_score:
        return {**face_emotion, "source": "face"}
    if voice_score >= 0.72 and voice_score > face_score:
        return {**voice_emotion, "source": "voice"}

    return {
        "label": face_label if face_score >= voice_score else voice_label,
        "score": float(max(face_score, voice_score)),
        "source": "mixed",
    }


def build_song_payload(song: Any) -> Dict[str, Any]:
    youtube_url = song.urls.get("youtube")
    return {
        "title": song.title,
        "artist": song.artist,
        "emotion": song.emotion,
        "spotify": song.urls.get("spotify"),
        "youtubeUrl": youtube_url,
        "youtubeEmbedId": _extract_youtube_video_id(youtube_url),
    }


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    payload = request.get_json(silent=True) or {}
    photo = payload.get("photo")
    audio = payload.get("audio")

    face_result = detect_face_emotion(photo) if photo else None
    voice_result = detect_voice_emotion(audio) if audio else None
    final_result = combine_emotions(face_result, voice_result)

    try:
        songs = recommender.recommend(final_result["label"], limit=5)
    except Exception:
        songs = recommender.recommend("neutral", limit=5)

    response = {
        "emotion": final_result["label"],
        "confidence": round(float(final_result.get("score", 0.0)), 2),
        "source": final_result.get("source", "mixed"),
        "facePrediction": face_result,
        "voicePrediction": voice_result,
        "songs": [build_song_payload(song) for song in songs],
    }
    return jsonify(response)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
