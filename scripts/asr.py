"""Pick the ASR backend the pipeline uses to locate clips (and to draft text for recordings with no log).

  ASR_BACKEND=server  (default)  voicera_asr.py: VoicEra / OpenAI-style ASR server over HTTP
  ASR_BACKEND=local              local_nemo_asr.py: a NeMo .nemo checkpoint loaded on this machine's GPU

Both backends expose the same interface:
  load_16k(path, start=None, end=None) -> float32 mono 16 kHz array
  decode_batch(crops) -> [text, ...]     (in input order)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402,F401  (loads .env)

BACKEND = os.environ.get("ASR_BACKEND", "server").lower()
if BACKEND == "local":
    from local_nemo_asr import SR, load_16k, decode_batch  # noqa: E402,F401
elif BACKEND == "server":
    from voicera_asr import SR, load_16k, decode_batch  # noqa: E402,F401
else:
    raise SystemExit(f"ASR_BACKEND must be 'server' or 'local', not {BACKEND!r}")
