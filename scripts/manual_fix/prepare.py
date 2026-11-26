"""Prepare the manual-fix queue: farmer turns with a recording but no clip (align_status=unaligned).

For each one, cut a listening window from the recording: from the start of the previous
aligned turn (bot or farmer) to the end of the next one, i.e. the stretch where the farmer's
speech must be if it was recorded at all. Windows are capped at MAX_WIN s and padded.
The best automatic candidate is pre-selected: the best diarized farmer stretch from
work/diar_located.csv if there is one, else the sub-crop search guess from work/recovered.csv.
Diarized farmer/bot time ranges (work/diar/) are included so the page can show them.

Writes work/manual/items.json and work/manual/audio/<utt_id>.wav (16 kHz PCM16; browsers
can't play the MPEG-in-WAV originals).

Run in the bhili-asr env: python scripts/manual_fix/prepare.py
"""
import csv, json, os, sys, collections
import librosa, soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from align import align_call  # noqa: E402

SR, MAX_WIN, PAD, CTX = 16000, 45.0, 2.0, 2


def window(ts, segs, assign, i, dur, guess_end=None):
    """Listening window for turn index i: start of the previous aligned turn's span to the end
    of the next one's, capped at MAX_WIN (kept around guess_end if given) and padded.
    Shared with diar_place.py."""
    prev = [k for k in assign if k < i]
    nxt = [k for k in assign if k > i]
    w0 = segs[assign[max(prev)][0]]["s"] if prev else 0.0
    w1 = segs[assign[min(nxt)][1] - 1]["e"] if nxt else dur
    if w1 - w0 > MAX_WIN:
        if guess_end is not None and guess_end > w0 + MAX_WIN:
            w0 = max(w0, guess_end - MAX_WIN + PAD)
        w1 = w0 + MAX_WIN
    return max(0.0, w0 - PAD), min(dur, w1 + PAD)


def main():
    os.chdir(ROOT)
    turns = collections.defaultdict(list)
    for r in csv.DictReader(open("work/turns.csv", encoding="utf-8")):
        turns[r["sid8"]].append(r)
    utt = [u for u in csv.DictReader(open("utterances.csv", encoding="utf-8-sig")) if u["align_status"] == "unaligned"]
    rec = {(r["sid8"], r["turn_idx"]): r for r in csv.DictReader(open("work/recovered.csv", encoding="utf-8"))}
    # best diarized farmer stretch per turn (diar_place.py), preferred as the pre-selected guess
    if os.path.exists("work/diar_located.csv"):
        for r in csv.DictReader(open("work/diar_located.csv", encoding="utf-8")):
            if r["start"] and r["region_asr"].strip():
                rec[(r["sid8"], r["turn_idx"])] = {"start": r["start"], "end": r["end"], "crop_asr": r["region_asr"],
                                                   "reject_reason": f"best diarized farmer stretch, text match {float(r['match_ratio']):.2f}"}
    os.makedirs("work/manual/audio", exist_ok=True)
    items, by_call = [], collections.defaultdict(list)
    for u in utt:
        by_call[u["sid8"]].append(u)
    for sid, us in by_call.items():
        d = json.load(open(f"work/segments/{sid}.json", encoding="utf-8"))
        segs, ts = d["segments"], turns[sid]
        assign = align_call(ts, segs)
        y = librosa.load(d["audio"], sr=SR, mono=True)[0]
        dpath = f"work/diar/{sid}.json"
        diar = json.load(open(dpath))["segments"] if os.path.exists(dpath) else []
        dur = len(y) / SR
        for u in us:
            i = int(u["turn_idx"]) - 1
            r = rec.get((sid, u["turn_idx"]))
            w0, w1 = window(ts, segs, assign, i, dur, float(r["end"]) if r else None)
            sf.write(f"work/manual/audio/{u['utt_id']}.wav", y[int(w0 * SR):int(w1 * SR)], SR, subtype="PCM_16")
            ctx = [{"turn": int(t["turn_idx"]), "speaker": "farmer" if t["speaker"] == "user" else "bot",
                    "text": t["text"] if len(t["text"]) < 160 else t["text"][:157] + "…", "this": t["turn_idx"] == u["turn_idx"]}
                   for t in ts[max(0, i - CTX):i + CTX + 1]]
            items.append({
                "utt_id": u["utt_id"], "sid8": sid, "call_folder": u["call_folder"], "turn_idx": int(u["turn_idx"]),
                "log_text": u["production_asr"], "audio": f"audio/{u['utt_id']}.wav",
                "win_start": round(w0, 3), "win_end": round(w1, 3), "source_recording": u["source_recording"],
                "guess_start": round(float(r["start"]) - w0, 2) if r else None,
                "guess_end": round(float(r["end"]) - w0, 2) if r else None,
                "guess_asr": r["crop_asr"] if r else "", "guess_note": r["reject_reason"] if r else "",
                "context": ctx,
                # where the diarizer heard each role inside the window (seconds from window start)
                "farmer_spans": [[round(max(0, x["s"] - w0), 2), round(min(w1, x["e"]) - w0, 2)] for x in diar
                                 if x["role"] == "farmer" and x["e"] > w0 and x["s"] < w1],
                "bot_spans": [[round(max(0, x["s"] - w0), 2), round(min(w1, x["e"]) - w0, 2)] for x in diar
                              if x["role"] == "bot" and x["e"] > w0 and x["s"] < w1],
            })
    items.sort(key=lambda x: (x["call_folder"], x["sid8"], x["turn_idx"]))
    json.dump(items, open("work/manual/items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(items)} items, {sum(x['guess_start'] is not None for x in items)} with a pre-selected guess, "
          f"{sum(x['win_end'] - x['win_start'] for x in items) / 60:.1f} min of window audio")


if __name__ == "__main__":
    main()
