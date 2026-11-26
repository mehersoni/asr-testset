"""ASR backend "server": a VoicEra (or any OpenAI-style) ASR server. Selected by scripts/asr.py
when ASR_BACKEND=server (the default). Interface: load_16k, decode_batch.

Server settings come from the environment, else from the .env file at the repo/test-set root
(or the file named by ASR_ENV_FILE): ASR_BASE_URL, ASR_API_KEY (optional), ASR_MODEL (default
indic-nemotron), ASR_LANG (default mr). Requests go to POST {ASR_BASE_URL}/v1/audio/transcriptions
(non-streaming, whole crop). Never commit the .env file.

Every result is cached in work/voicera_cache.jsonl, keyed by a hash of the 16 kHz PCM16 crop, so
re-runs only send crops the server has not seen.
"""
import hashlib, io, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402  (loads .env into the environment)

SR = 16000

ROOT, ENV_FILE = config.ROOT, config.ENV_FILE
CACHE = os.environ.get("ASR_CACHE") or os.path.join(ROOT, "work", "voicera_cache.jsonl")
# One request at a time: in Nov 2026 the VoicEra server returned different, garbled text for some crops
# when sent 8 concurrent requests (2 of 16 clips), while sequential requests were deterministic.
# Raise ASR_WORKERS only after checking that your server gives the same text under concurrency.
WORKERS = int(os.environ.get("ASR_WORKERS", "1"))
MIN_SAMPLES = int(0.1 * SR)       # shorter crops are returned as "" without a request


BASE_URL = os.environ.get("ASR_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("ASR_API_KEY", "")
MODEL_ID = os.environ.get("ASR_MODEL") or "indic-nemotron"
LANG = os.environ.get("ASR_LANG") or "mr"
if not BASE_URL:
    raise SystemExit(f"no ASR_BASE_URL in the environment or {ENV_FILE}")
_headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
_lock = threading.Lock()
_cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE, encoding="utf-8"):
        try:
            r = json.loads(line)
            _cache[r["k"]] = r["t"]
        except (ValueError, KeyError):
            pass
print(f"[voicera_asr] {BASE_URL} model={MODEL_ID} lang={LANG}, {len(_cache)} cached", flush=True)


def load_16k(path, start=None, end=None):
    """Load mono audio resampled to 16 kHz, optionally cropped to [start, end] seconds."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if start is not None or end is not None:
        audio = audio[int((start or 0) * SR): len(audio) if end is None else int(end * SR)]
    return audio.astype(np.float32)


def _pcm16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _request(pcm):
    buf = io.BytesIO()
    sf.write(buf, np.frombuffer(pcm, dtype="<i2"), SR, format="WAV", subtype="PCM_16")
    for attempt in range(6):
        try:
            r = requests.post(f"{BASE_URL}/v1/audio/transcriptions", headers=_headers, timeout=120,
                              files={"file": ("crop.wav", buf.getvalue(), "audio/wav")},
                              data={"model": MODEL_ID, "language": LANG})
            if r.status_code == 200:
                return (r.json().get("text") or "").strip()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"VoicEra ASR failed after retries: {err}")


def _one(audio):
    if len(audio) < MIN_SAMPLES:
        return ""
    pcm = _pcm16(audio)
    k = hashlib.sha1(pcm).hexdigest()
    with _lock:
        if k in _cache:
            return _cache[k]
    t = _request(pcm)
    with _lock:
        _cache[k] = t
        with open(CACHE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "t": t}, ensure_ascii=False) + "\n")
    return t


def decode_batch(crops):
    """Decode a list of 1-D float32 16 kHz arrays; results in input order."""
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(_one, [np.asarray(c, dtype=np.float32) for c in crops]))
