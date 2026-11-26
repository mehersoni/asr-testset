"""Step 2f: trim clips that are mostly bot speech down to the farmer's speech inside them.

Why: farmers often barge in during a long bot answer (in the Marathi calls: "हो हो",
"हॅलो"). VAD keeps bot + farmer as one long segment, and align.py accepts it because the
farmer's words do appear in its ASR, so the clip can be 20-47 s of bot audio around a 1 s "हो".

For each included clip (production_log rows) with diar_farmer_share < MAX_SHARE and at
least MIN_NONFARMER s of non-farmer audio:
  candidates = runs of consecutive farmer-labelled diarized stretches (work/diar/<sid8>.json,
               merged across gaps < MERGE_GAP s) inside the clip, up to MAX_RUN stretches long and
               with at most MAX_INNER_GAP s of non-farmer audio between them (a logged turn that is
               really two bursts around a bot answer keeps its long clip and is flagged on export)
  each candidate (padded by PAD) is decoded with the ASR backend and scored against the logged
  text (user_score from recover_overlap.py)
accept the best candidate if match_ratio >= ACCEPT and it scores at least as well as the whole
clip. Rejected clips are left as they are (export flags them low_farmer_speech / overlap_with_bot).

The untrimmed span is kept in orig_start/orig_end, so a re-run (after build_testset.py has
applied the trims) starts from the original span again.

Writes work/diar_trimmed.csv. Run in the bhili-asr env, after build_testset.py has run once:
  CUDA_VISIBLE_DEVICES=0 python scripts/diar_trim.py && python scripts/build_testset.py
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from align import ratio  # noqa: E402
from recover_overlap import decode_batch, user_score  # noqa: E402  (ASR backend: scripts/asr.py)
import librosa  # noqa: E402

SR, BUILD_PAD = 16000, 0.10      # build_testset.py pads each span by BUILD_PAD
MAX_SHARE, MIN_NONFARMER = 0.6, 2.0
MERGE_GAP, MAX_RUN, PAD, ACCEPT = 0.4, 6, 0.15, 0.5
MAX_INNER_GAP = 2.5             # non-farmer audio a trimmed clip may still contain (s)
OUT = "work/diar_trimmed.csv"


def main():
    os.chdir(ROOT)
    utt = list(csv.DictReader(open("utterances.csv", encoding="utf-8-sig")))
    old = {r["utt_id"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"))} if os.path.exists(OUT) else {}
    rows, cache = [], {}
    for u in utt:
        if u["include"] != "yes" or u["transcript_source"] != "production_log" or u["diar_farmer_share"] == "":
            continue
        uid, sid = u["utt_id"], u["sid8"]
        if uid in old:      # already trimmed by an earlier run: start from the original span
            s0, e0 = float(old[uid]["orig_start"]), float(old[uid]["orig_end"])
        else:
            s0, e0 = float(u["rec_start_s"]) + BUILD_PAD, float(u["rec_end_s"]) - BUILD_PAD
        dj = json.load(open(f"work/diar/{sid}.json"))
        far = []
        for x in sorted((x for x in dj["segments"] if x["role"] == "farmer"), key=lambda x: x["s"]):
            a, b = max(s0, x["s"]), min(e0, x["e"])
            if b <= a:
                continue
            if far and a - far[-1][1] < MERGE_GAP:
                far[-1][1] = max(far[-1][1], b)
            else:
                far.append([a, b])
        farmer_s = sum(b - a for a, b in far)
        dur = e0 - s0
        if farmer_s / max(dur, 1e-6) >= MAX_SHARE or dur - farmer_s < MIN_NONFARMER:
            continue
        row = {"utt_id": uid, "sid8": sid, "turn_idx": u["turn_idx"], "orig_start": round(s0, 3),
               "orig_end": round(e0, 3), "start": "", "end": "", "trim_asr": "", "match_ratio": "",
               "orig_asr": "", "accepted": "", "reason": ""}
        rows.append(row)
        if not far:
            row["reason"] = "no farmer speech in the clip"
            continue
        audio = u["source_recording"]
        if audio not in cache:
            cache.clear()
            cache[audio] = librosa.load(os.path.join("raw", audio), sr=SR, mono=True)[0]
        y = cache[audio]
        spans = [(s0, e0)]
        for i in range(len(far)):
            for j in range(i, min(len(far), i + MAX_RUN)):
                if sum(far[k + 1][0] - far[k][1] for k in range(i, j)) > MAX_INNER_GAP:
                    break
                spans.append((max(s0, far[i][0] - PAD), min(e0, far[j][1] + PAD)))
        if len(spans) == 1:
            row["reason"] = "farmer speech too short to decode"
            continue
        texts = decode_batch([y[int(a * SR):int(b * SR)] for a, b in spans])
        log = u["production_asr"]
        orig_score = user_score(log, texts[0])
        best = max(range(1, len(spans)), key=lambda k: (user_score(log, texts[k]), -(spans[k][1] - spans[k][0])))
        r = round(ratio(log, texts[best]), 3)
        row.update(start=round(spans[best][0], 3), end=round(spans[best][1], 3), trim_asr=texts[best],
                   match_ratio=r, orig_asr=texts[0])
        if r < ACCEPT:
            row["reason"] = f"match_ratio < {ACCEPT}"
        elif user_score(log, texts[best]) < orig_score:
            row["reason"] = "whole clip matches the log better"
        else:
            row["accepted"] = "yes"
        print(f"{uid} {dur:.1f}s -> {spans[best][1] - spans[best][0]:.1f}s r={r} {row['accepted'] or row['reason']} | "
              f"{log[:50]} | {texts[best][:50]}", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=['utt_id', 'sid8', 'turn_idx', 'orig_start', 'orig_end', 'start', 'end', 'trim_asr', 'match_ratio', 'orig_asr', 'accepted', 'reason']); w.writeheader(); w.writerows(rows)  # header even if empty
    acc = [r for r in rows if r["accepted"]]
    saved = sum((r["orig_end"] - r["orig_start"]) - (r["end"] - r["start"]) for r in acc)
    print(f"{len(rows)} mostly-bot clips, {len(acc)} trimmed (-{saved / 60:.1f} min of non-farmer audio) -> {OUT}")


if __name__ == "__main__":
    main()
