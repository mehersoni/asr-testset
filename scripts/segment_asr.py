"""Step 1 of alignment: VAD-segment every call recording (Silero VAD) and ASR each segment
with the backend chosen in scripts/asr.py (ASR_BACKEND=server|local).

Run in the bhili-asr env (Python 3.11, NeMo):
  python scripts/segment_asr.py raw/recordings work/turns.csv work/segments

Writes work/segments/<sid8>.json = {"audio": path, "duration": s, "segments": [{"s","e","text"}]}.
Calls with no transcript are skipped. Existing outputs are reused.
"""
import csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr  # noqa: E402  (ASR backend: scripts/asr.py)
from silero_vad import get_speech_timestamps, load_silero_vad  # noqa: E402
import torch  # noqa: E402

SR = 16000


def sid8_of(path):
    name = os.path.basename(path)
    m = re.match(r"([0-9a-f]{8})-", name)
    return m.group(1) if m else re.findall(r"[0-9a-f]{8}", name)[-1]


def main(rec_root, turns_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wanted = {r["sid8"] for r in csv.DictReader(open(turns_csv, encoding="utf-8"))}
    vad = load_silero_vad()
    wavs = sorted(glob.glob(os.path.join(rec_root, "**", "*.wav"), recursive=True))
    for i, w in enumerate(wavs, 1):
        sid = sid8_of(w)
        out = os.path.join(out_dir, f"{sid}.json")
        if sid not in wanted or os.path.exists(out):
            continue
        audio = asr.load_16k(w)
        ts = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR,
                                   min_silence_duration_ms=250, speech_pad_ms=120,
                                   min_speech_duration_ms=150, max_speech_duration_s=20)
        texts = asr.decode_batch([audio[t["start"]:t["end"]] for t in ts])
        segs = [{"s": round(t["start"] / SR, 3), "e": round(t["end"] / SR, 3), "text": x} for t, x in zip(ts, texts)]
        json.dump({"audio": w, "duration": round(len(audio) / SR, 3), "segments": segs},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(f"[{i}/{len(wavs)}] {sid} {len(segs)} segs", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
