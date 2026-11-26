"""Step 2e: place still-unaligned farmer turns on farmer speech found by diarization.

For each farmer turn with align_status=unaligned (the manual-fix queue):
  candidates = farmer-labelled diarized stretches (work/diar/<sid8>.json, merged across gaps
               < MERGE_GAP s) that overlap the turn's listening window (manual_fix/prepare.window,
               with the sub-crop guess from work/recovered.csv, as the listening tool first used)
               and are not already covered by an existing clip; plus pairs of neighbouring
               candidates (gap < PAIR_GAP s), for a sentence broken by a pause
  each candidate is decoded with the ASR backend and scored against the logged text
  (same score as recover_overlap.py); turns are placed in conversation order, so a turn can't
  take a stretch earlier than the previous placed turn, and stretches aren't shared
accept if match_ratio >= ACCEPT.

Also reports farmer stretches that no logged turn claims (farmer speech production never logged).

Writes work/diar_located.csv. Run in the bhili-asr env:
  CUDA_VISIBLE_DEVICES=0 python scripts/diar_place.py
"""
import csv, json, os, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from align import align_call, ratio  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "manual_fix"))
from prepare import window  # noqa: E402
from recover_overlap import decode_batch, user_score  # noqa: E402  (ASR backend: scripts/asr.py)
import librosa  # noqa: E402

SR, MERGE_GAP, PAIR_GAP, ACCEPT, PAD = 16000, 0.4, 1.5, 0.5, 0.1
INCLUDED = ("ok", "check", "overlap_recovered", "crop_recovered", "manual_located")   # not diar_located


def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    os.chdir(ROOT)
    utt = list(csv.DictReader(open("utterances.csv", encoding="utf-8-sig")))
    turns = collections.defaultdict(list)
    for r in csv.DictReader(open("work/turns.csv", encoding="utf-8")):
        turns[r["sid8"]].append(r)
    guess = {(r["sid8"], r["turn_idx"]): float(r["end"]) for r in csv.DictReader(open("work/recovered.csv", encoding="utf-8"))}
    by = collections.defaultdict(list)
    for u in utt:
        by[u["sid8"]].append(u)
    rows, unclaimed = [], []
    for sid, us in sorted(by.items()):
        # diar_located rows are this script's own earlier output: redo them too, so a re-run is stable
        todo = [u for u in us if u["align_status"] in ("unaligned", "diar_located")]
        if not todo or not os.path.exists(f"work/diar/{sid}.json"):
            continue
        dj = json.load(open(f"work/diar/{sid}.json"))
        # merge farmer stretches across short gaps
        far = []
        for x in [x for x in dj["segments"] if x["role"] == "farmer"]:
            if far and x["s"] - far[-1][1] < MERGE_GAP:
                far[-1][1] = max(far[-1][1], x["e"])
            else:
                far.append([x["s"], x["e"]])
        bot = [(x["s"], x["e"]) for x in dj["segments"] if x["role"] == "bot"]
        clips = [(float(u["rec_start_s"]), float(u["rec_end_s"])) for u in us if u["align_status"] in INCLUDED]
        free = [(s, e) for s, e in far if not any(ov(s, e, a, b) > 0.5 * (e - s) for a, b in clips)]
        y = librosa.load(dj["audio"], sr=SR, mono=True)[0]
        dur = len(y) / SR
        sj = json.load(open(f"work/segments/{sid}.json", encoding="utf-8"))
        assign = align_call(turns[sid], sj["segments"])
        cands = {}
        for u in todo:
            w0, w1 = window(turns[sid], sj["segments"], assign, int(u["turn_idx"]) - 1, dur, guess.get((sid, u["turn_idx"])))
            idx = [k for k, (s, e) in enumerate(free) if ov(s, e, w0, w1) > 0]
            spans = [free[k] for k in idx] + [(free[a][0], free[b][1]) for a, b in zip(idx, idx[1:])
                                               if b == a + 1 and free[b][0] - free[a][1] < PAIR_GAP]
            cands[u["utt_id"]] = [(round(max(0, s - PAD), 2), round(min(dur, e + PAD), 2)) for s, e in spans]
        uniq = sorted({c for v in cands.values() for c in v})
        texts = dict(zip(uniq, decode_batch([y[int(s * SR):int(e * SR)] for s, e in uniq]))) if uniq else {}
        taken, last_end = [], 0.0
        for u in sorted(todo, key=lambda u: int(u["turn_idx"])):
            best = None
            for s, e in cands[u["utt_id"]]:
                if s < last_end - 0.5 or any(ov(s, e, a, b) > 0.2 for a, b in taken):
                    continue
                sc = user_score(u["production_asr"], texts[(s, e)])
                if best is None or sc > best[0] or (sc == best[0] and e - s < best[2] - best[1]):
                    best = (sc, s, e, texts[(s, e)])
            row = {"utt_id": u["utt_id"], "sid8": sid, "turn_idx": u["turn_idx"], "log_text": u["production_asr"],
                   "n_candidates": len(cands[u["utt_id"]]), "start": "", "end": "", "region_asr": "",
                   "match_ratio": "", "overlaps_bot": "", "accepted": "", "reason": ""}
            if best is None:
                row["reason"] = "no free farmer speech in the window"
            else:
                _, s, e, t = best
                r = ratio(u["production_asr"], t)
                row.update(start=s, end=e, region_asr=t, match_ratio=r,
                           overlaps_bot="yes" if sum(ov(s, e, a, b) for a, b in bot) > 0.3 else "")
                if r >= ACCEPT:
                    row["accepted"] = "yes"
                    taken.append((s, e)); last_end = e
                else:
                    row["reason"] = f"best farmer stretch matches poorly (ratio {r:.2f})"
            rows.append(row)
        # farmer stretches nobody claims
        claimed = clips + taken
        unclaimed += [(sid, s, e) for s, e in far if not any(ov(s, e, a, b) > 0.3 * (e - s) for a, b in claimed)]
        print(f"{sid}: {sum(r['accepted'] == 'yes' for r in rows if r['sid8'] == sid)}/{len(todo)} placed", flush=True)
    with open("work/diar_located.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=['utt_id', 'sid8', 'turn_idx', 'log_text', 'n_candidates', 'start', 'end', 'region_asr', 'match_ratio', 'overlaps_bot', 'accepted', 'reason']); w.writeheader(); w.writerows(rows)  # header even if empty
    acc = [r for r in rows if r["accepted"]]
    print(f"PLACED {len(acc)}/{len(rows)} ({sum(float(r['end']) - float(r['start']) for r in acc) / 60:.1f} min), "
          f"over bot speech: {sum(bool(r['overlaps_bot']) for r in acc)}")
    print("NOT PLACED:", dict(collections.Counter(r["reason"].split(" (")[0] for r in rows if not r["accepted"])))
    long_unclaimed = [x for x in unclaimed if x[2] - x[1] >= 1.0]
    print(f"UNCLAIMED farmer stretches >=1 s: {len(long_unclaimed)}, {sum(e - s for _, s, e in long_unclaimed) / 60:.1f} min")


if __name__ == "__main__":
    main()
