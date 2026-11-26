"""ASR backend "local": a NeMo indic-nemotron checkpoint loaded in-process (selected by scripts/asr.py
when ASR_BACKEND=local). This is how the Bhili test set was built (Bhili Nemotron SFT, prompt bhb).

Loading goes through VoicEra's production loader, model-server/stt/indic-nemotron/asr_engine.py
(https://github.com/COSS-India/VoicEra): it drops the multisoftmax kwargs NeMo 3.0.0 rejects and
slices the output layer to the language head. Decoding is one offline pass per crop through
conformer_stream_step (same encoder context as streaming, no chunking), batched.

Environment:
  VOICERA_ENGINE_DIR   path to VoicEra/model-server/stt/indic-nemotron            (required)
  LOCAL_ASR_MODEL      bhili (default) or indic: which of asr_engine's two models to use
  BHILI_NEMO_PATH      .nemo file for the Bhili model   } asr_engine.py reads these; set the one
  INDIC_NEMO_PATH      .nemo file for the indic model   } for LOCAL_ASR_MODEL
  ASR_LANG             language prompt (default bhb). Check it on a few clips first: the Bhili SFT's
                       `mr` prompt returns empty text.
  ASR_ATT_CONTEXT      encoder attention context "left,right" (default 96,7)
  ASR_BATCH            crops per GPU batch (default 64)

Needs Python >= 3.11 and NeMo (the bhili-asr conda env) and a GPU.
"""
import os
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402,F401  (loads .env)

SR = 16000
ENGINE_DIR = os.environ.get("VOICERA_ENGINE_DIR")
if not ENGINE_DIR:
    raise SystemExit("ASR_BACKEND=local needs VOICERA_ENGINE_DIR (VoicEra/model-server/stt/indic-nemotron)")
WHICH = os.environ.get("LOCAL_ASR_MODEL", "bhili")
LANG = os.environ.get("ASR_LANG", "bhb")
ATT = tuple(int(x) for x in os.environ.get("ASR_ATT_CONTEXT", "96,7").split(","))
BATCH = int(os.environ.get("ASR_BATCH", "64"))

# load only the model we need
os.environ.setdefault("INDIC_NEMO_PATH" if WHICH == "bhili" else "BHILI_NEMO_PATH", "/nonexistent")
sys.path.insert(0, ENGINE_DIR)
import asr_engine  # noqa: E402  (loads the model at import)

MODEL = asr_engine.bhili_model if WHICH == "bhili" else asr_engine.indic_model
if MODEL is None:
    raise SystemExit(f"asr_engine could not load the {WHICH} model; set {WHICH.upper()}_NEMO_PATH")
MODEL.encoder.set_default_att_context_size(att_context_size=list(ATT))
asr_engine.set_prompt(MODEL, LANG)
print(f"[local_nemo_asr] model={WHICH} prompt={LANG} att={ATT}", flush=True)


def load_16k(path, start=None, end=None):
    """Load mono audio resampled to 16 kHz, optionally cropped to [start, end] seconds."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if start is not None or end is not None:
        audio = audio[int((start or 0) * SR): len(audio) if end is None else int(end * SR)]
    return audio.astype(np.float32)


def _text(h):
    return ((h.text if hasattr(h, "text") else h) or "").strip()


@torch.inference_mode()
def decode_batch(crops):
    """Offline decode of a list of 1-D float32 16 kHz arrays; results in input order.
    Crops shorter than 0.1 s come back as "" (same as the server backend)."""
    crops = [np.asarray(c, dtype=np.float32) for c in crops]
    keep = [k for k, c in enumerate(crops) if len(c) >= int(0.1 * SR)]
    res = [""] * len(crops)
    for k, t in zip(keep, _decode([crops[k] for k in keep])):
        res[k] = t
    return res


def _decode(crops):
    out = []
    for i in range(0, len(crops), BATCH):
        chunk = crops[i:i + BATCH]
        n = max(len(c) for c in chunk)
        sig = torch.zeros(len(chunk), n, device=MODEL.device)
        for k, c in enumerate(chunk):
            sig[k, :len(c)] = torch.from_numpy(c)
        ln = torch.tensor([len(c) for c in chunk], device=MODEL.device)
        feats, flen = MODEL.preprocessor(input_signal=sig, length=ln)
        _, texts, *_ = MODEL.conformer_stream_step(processed_signal=feats, processed_signal_length=flen,
                                                   return_transcription=True)
        out += [_text(h) for h in texts]
    return out

