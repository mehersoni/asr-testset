"""Step 2b: recover farmer turns that the segment alignment missed, typically speech that
overlaps the bot (barge-in).

Why: VAD puts overlapped farmer speech in the same segment as the bot, and the ASR of the
whole segment follows the bot. Decoding only the part where the farmer speaks does recover
it (in one Bhili call, a 5 s segment decoded as the bot's line, while its last 4 s decoded as
the farmer's question).

For each farmer turn still unaligned after align.py (and with >= MIN_CHARS of logged text):
  search window = start of the previous aligned turn's span .. end of the next aligned turn's span
                  (capped at MAX_WIN s), i.e. including the neighbouring bot audio it may overlap
  coarse: decode fixed-length crops (COARSE_LENS) every COARSE_HOP s across the window
  refine: move the start (anywhere inside the crop, or up to FINE_RANGE earlier), then the end,
          then both together at FINE_STEP
  score:  2*LCS - 0.5*unmatched log chars - 1.0*unmatched ASR chars (bot speech in the crop costs more)
  accept if char similarity (match_ratio) >= ACCEPT, the crop does not overlap another
  farmer clip, and the bot's own text doesn't explain the crop ASR as well as the farmer's.

Run in the bhili-asr env:
  python scripts/recover_overlap.py work/turns.csv work/segments work/alignment.csv work/recovered.csv
"""
import csv, json, os, sys, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from align import norm, lcs, ratio  # noqa: E402
import asr  # noqa: E402  (ASR backend: scripts/asr.py)

SR = 16000
MIN_CHARS = 4
MAX_WIN = 45.0
COARSE_LENS = (1.5, 3.0, 5.0, 8.0)
COARSE_HOP = 0.5
FINE_STEP, FINE_RANGE = 0.1, 1.0
ACCEPT = 0.5
BATCH = 64


def decode_batch(crops):
    """Decode a list of 1-D float32 16 kHz arrays with the configured ASR backend (scripts/asr.py)."""
    return asr.decode_batch(crops)


def user_score(log_txt, asr_txt):
    """Reward farmer chars found; penalise unmatched log chars (0.5) and, harder, ASR chars the
    log doesn't explain (1.0), so crops that drag in bot speech lose to tighter ones."""
    a, b = norm(log_txt), norm(asr_txt)
    m = lcs(a, b)
    return 2 * m - 0.5 * (len(a) - m) - 1.0 * (len(b) - m)


def _best(audio, log_txt, spans):
    spans = [(round(float(s), 2), round(float(e), 2)) for s, e in spans if e - s >= 0.4]
    texts = decode_batch([audio[int(s * SR):int(e * SR)] for s, e in spans])
    cands = list(zip(spans, texts))
    top = max(user_score(log_txt, t) for _, t in cands)
    # among equal scores take the shortest crop
    return min([c for c in cands if user_score(log_txt, c[1]) == top], key=lambda c: c[0][1] - c[0][0])


def search(audio, log_txt, w0, w1):
    dur = len(audio) / SR
    w0, w1 = max(0.0, w0), min(dur, w1)
    spans = [(s, min(s + L, w1)) for L in COARSE_LENS for s in np.arange(w0, max(w0, w1 - L) + 1e-6, COARSE_HOP)]
    if not [1 for s, e in spans if e - s >= 0.4]:
        return None
    (s, e), t = _best(audio, log_txt, spans)
    # refine: start anywhere from 1 s before to just before the end, then the end likewise
    (s, e), t = _best(audio, log_txt, [(x, e) for x in np.arange(max(w0, s - FINE_RANGE), e - 0.4, 0.2)] + [(s, e)])
    (s, e), t = _best(audio, log_txt, [(s, x) for x in np.arange(s + 0.4, min(w1, e + FINE_RANGE) + 1e-6, 0.2)] + [(s, e)])
    # final joint pass at 0.1 s
    (s, e), t = _best(audio, log_txt, [(a, b) for a in np.arange(max(w0, s - 0.5), s + 0.5 + 1e-6, FINE_STEP)
                                       for b in np.arange(e - 0.5, min(w1, e + 0.5) + 1e-6, FINE_STEP)] + [(s, e)])
    return s, e, t


def main(turns_csv, seg_dir, align_csv, out_csv):
    turns = collections.defaultdict(list)
    for r in csv.DictReader(open(turns_csv, encoding="utf-8")):
        turns[r["sid8"]].append(r)
    al = {(r["sid8"], r["turn_idx"]): r for r in csv.DictReader(open(align_csv, encoding="utf-8"))}
    from align import align_call
    only = set(filter(None, os.environ.get("SIDS", "").split(",")))   # optional: restrict to these calls
    rows = []
    for sid, ts in turns.items():
        if only and sid not in only:
            continue
        f = os.path.join(seg_dir, f"{sid}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        segs = d["segments"]
        assign = align_call(ts, segs)           # same DP as align.py -> spans of every aligned turn
        todo = [i for i, t in enumerate(ts) if t["speaker"] == "user" and i not in assign
                and len(norm(t["text"])) >= MIN_CHARS]
        if not todo:
            continue
        audio = asr.load_16k(d["audio"])
        taken = [(segs[a]["s"], segs[b - 1]["e"]) for i, (a, b) in assign.items() if ts[i]["speaker"] == "user"]
        bot_texts = [t["text"] for t in ts if t["speaker"] == "assistant"]
        bot_spans = [(segs[a]["s"], segs[b - 1]["e"]) for i, (a, b) in assign.items() if ts[i]["speaker"] == "assistant"]
        for i in todo:
            prev = [k for k in assign if k < i]
            nxt = [k for k in assign if k > i]
            w0 = segs[assign[max(prev)][0]]["s"] if prev else 0.0
            w1 = segs[assign[min(nxt)][1] - 1]["e"] if nxt else len(audio) / SR
            if w1 - w0 > MAX_WIN:                # keep the part nearest the previous anchor
                w1 = w0 + MAX_WIN
            res = search(audio, ts[i]["text"], w0, w1)
            if not res:
                continue
            s, e, txt = res
            r = ratio(ts[i]["text"], txt)
            clash = any(s < te and e > tsx for tsx, te in taken)
            over_bot = any(s < be and e > bs for bs, be in bot_spans)
            # the farmer may repeat the bot's words (crop names...): reject crops the bot's own
            # text explains at least as well as the farmer's logged text
            y = norm(txt)
            botlike = max((lcs(y, norm(b)) for b in bot_texts), default=0) >= lcs(y, norm(ts[i]["text"]))
            reason = ("overlaps another farmer clip" if clash else
                      f"match_ratio < {ACCEPT}" if r < ACCEPT else
                      "crop ASR matches bot text as well as farmer text" if botlike else "")
            ok = not reason
            if ok:
                taken.append((s, e))
            rows.append({"sid8": sid, "turn_idx": ts[i]["turn_idx"], "log_text": ts[i]["text"],
                         "win_start": round(w0, 2), "win_end": round(w1, 2), "start": s, "end": e,
                         "crop_asr": txt, "match_ratio": r, "overlaps_bot": "yes" if over_bot else "",
                         "accepted": "yes" if ok else "", "reject_reason": reason})
            print(f"{sid} t{ts[i]['turn_idx']} {r:.2f} {'ACC' if ok else '---'} {s}-{e} | {ts[i]['text'][:40]} | {txt[:40]}", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=['sid8', 'turn_idx', 'log_text', 'win_start', 'win_end', 'start', 'end', 'crop_asr', 'match_ratio', 'overlaps_bot', 'accepted', 'reject_reason']); w.writeheader(); w.writerows(rows)  # header even if empty
    print(f"{len(rows)} searched, {sum(r['accepted'] == 'yes' for r in rows)} recovered -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
