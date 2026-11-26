"""Step 2: align logged turns to VAD speech segments and pick each farmer turn's audio span.

The recordings are not on a linear clock with the call log (offset drifts by several
seconds within a call), so alignment uses order + text, not timestamps:

  monotonic DP over (log turns) x (speech segments). Each turn takes a contiguous run of
  0..K segments, or none; segments may be left unassigned (noise, unlogged speech).
  Score of turn<->segments = character LCS between the logged text and the segments' ASR:
    user turns:      2*LCS - 0.5*(len_log + len_asr)   (penalise unmatched chars on both sides)
    assistant turns: 1.5*LCS - 0.5*len_asr             (bot audio may be cut short by barge-in,
                                                          so unmatched log text is not penalised)
  Bot turns are long and match their TTS audio closely, so they act as anchors; farmer
  turns fall into the gaps between them.

Farmer turns the DP leaves unassigned are reported as "unaligned", not guessed. A timing
fallback (interpolating the log->audio offset) was tried and rejected: leave-one-out error
was only 60% within 1 s, and the turns it placed were mostly mispaired.

Usage: python scripts/align.py work/turns.csv work/segments work/alignment.csv
"""
import csv, glob, json, os, re, sys, collections
from rapidfuzz.distance import Indel

KU, KA = 4, 12          # max segments per user / assistant turn
SEG_COST = 0.05


def norm(t):
    return re.sub(r"[\s\.,!?।'\"\-:;()/]+", "", t or "")


def lcs(a, b):
    return (len(a) + len(b) - Indel.distance(a, b)) // 2 if a and b else 0


def score(speaker, log_txt, asr_txt):
    a, b = norm(log_txt), norm(asr_txt)
    m = lcs(a, b)
    if speaker == "user":
        return 2 * m - 0.5 * (len(a) + len(b))
    return 1.5 * m - 0.5 * len(b)


def align_call(turns, segs):
    n, m = len(turns), len(segs)
    NEG = float("-inf")
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = best[i][j]
            if cur == NEG:
                continue
            if j < m and cur > best[i][j + 1]:                       # leave segment j unassigned
                best[i][j + 1], back[i][j + 1] = cur, (i, j, "skipseg")
            if i < n:
                if cur > best[i + 1][j]:                             # turn i gets no audio
                    best[i + 1][j], back[i + 1][j] = cur, (i, j, "skipturn")
                t = turns[i]
                K = KU if t["speaker"] == "user" else KA
                txt = ""
                for k in range(j, min(m, j + K)):
                    txt += segs[k]["text"]
                    # small per-segment cost: on ties, prefer the tightest span (don't absorb
                    # neighbouring segments whose ASR is empty)
                    v = cur + score(t["speaker"], t["text"], txt) - SEG_COST * (k - j + 1)
                    if v > best[i + 1][k + 1]:
                        best[i + 1][k + 1], back[i + 1][k + 1] = v, (i, j, "take")
    # backtrack
    assign, i, j = {}, n, m
    while (i, j) != (0, 0):
        pi, pj, op = back[i][j]
        if op == "take":
            assign[pi] = (pj, j)          # segments pj..j-1
        i, j = pi, pj
    return assign


def ratio(a, b):
    x, y = norm(a), norm(b)
    return round(1 - Indel.normalized_distance(x, y), 3) if x or y else 0.0


def main(turns_csv, seg_dir, out_csv):
    calls = collections.defaultdict(list)
    for r in csv.DictReader(open(turns_csv, encoding="utf-8")):
        calls[r["sid8"]].append(r)
    rows = []
    for sid, turns in calls.items():
        f = os.path.join(seg_dir, f"{sid}.json")
        if not os.path.exists(f):
            for t in turns:
                if t["speaker"] == "user":
                    rows.append({**t, "audio": "", "seg_start": "", "seg_end": "", "n_segs": 0,
                                 "seg_asr": "", "match_ratio": "", "align_status": "no_recording"})
            continue
        d = json.load(open(f, encoding="utf-8"))
        segs = d["segments"]
        assign = align_call(turns, segs)
        for i, t in enumerate(turns):
            if t["speaker"] != "user":
                continue
            if i in assign:
                a, b = assign[i]
                asr = " ".join(s["text"] for s in segs[a:b])
                r = ratio(t["text"], asr)
                status = "ok" if r >= 0.6 else ("check" if r >= 0.35 else "weak")
                rows.append({**t, "audio": d["audio"], "seg_start": segs[a]["s"], "seg_end": segs[b - 1]["e"],
                             "n_segs": b - a, "seg_asr": asr, "match_ratio": r, "align_status": status})
            else:
                rows.append({**t, "audio": d["audio"], "seg_start": "", "seg_end": "", "n_segs": 0,
                             "seg_asr": "", "match_ratio": "", "align_status": "unaligned"})
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    c = collections.Counter(r["align_status"] for r in rows)
    print(f"{len(rows)} user turns: {dict(c)} -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
