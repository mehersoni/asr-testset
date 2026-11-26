"""Parse all call transcripts (call logs) into a flat turn table.

Handles the three layouts seen so far (Bhili and Marathi field tests):
  - plain:     "[ISO-ts] speaker: text"
  - headered:  "call_id=<uuid>\n---\n" then plain lines
  - numbered:  "<n>\t[ISO-ts] speaker: text"   (July 14)
July 14 also has _data_group_*.json with per-turn translation/observation notes,
which are joined on (session, turn number).
"""
import csv, glob, json, os, re, sys
from datetime import datetime

LINE = re.compile(r"^(?:\d+\t)?\[(\S+?)\] (user|assistant): ?(.*)$")
UUID8 = re.compile(r"([0-9a-f]{8})(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?")

def parse_file(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"call_id=([0-9a-f-]{36})", txt)
    name = os.path.basename(path)
    session = m.group(1) if m else None
    if not session:
        m2 = re.match(r"([0-9a-f-]{36})", name)
        session = m2.group(1) if m2 else None
    ids = UUID8.findall(name)
    sid8 = (session or (ids[-1] if ids else ""))[:8]
    turns, cur = [], None
    for raw in txt.splitlines():
        mm = LINE.match(raw)
        if mm:
            cur = {"ts": mm.group(1), "speaker": mm.group(2), "text": mm.group(3).strip()}
            turns.append(cur)
        elif cur and raw.strip() and not raw.startswith(("call_id=", "---")) and not re.fullmatch(r"\d+\t?\s*", raw):
            cur["text"] += " " + raw.strip()      # continuation line
    return session, sid8, turns

def main(roots, out_csv):
    """roots: transcript folders in priority order; a session seen in an earlier root is skipped later."""
    notes = {}
    for jf in [j for root in roots for j in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)]:
        for call in json.load(open(jf, encoding="utf-8")):
            for t in call["turns"]:
                notes[(call["sessionId"][:8], t["num"])] = t
    rows, seen = [], set()
    files = [(root, f) for root in roots for f in sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))]
    for root, f in files:
        session, sid8, turns = parse_file(f)
        if not turns:
            print("SKIP (no log-format turns):", f, file=sys.stderr); continue
        if sid8 in seen:
            continue
        seen.add(sid8)
        t0 = datetime.fromisoformat(turns[0]["ts"])
        for i, t in enumerate(turns, 1):
            n = notes.get((sid8, i), {})
            rows.append({
                "folder": os.path.basename(os.path.dirname(f)),
                "transcript_file": os.path.relpath(f, os.path.dirname(os.path.normpath(root))),
                "session_id": session or "", "sid8": sid8, "turn_idx": i, "ts_utc": t["ts"],
                "log_offset_s": round((datetime.fromisoformat(t["ts"]) - t0).total_seconds(), 3),
                "speaker": t["speaker"], "text": t["text"],
                "translation_en": n.get("translation", ""), "observation": n.get("observation", ""),
            })
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    u = [r for r in rows if r["speaker"] == "user"]
    print(f"{len(set(r['sid8'] for r in rows))} calls, {len(rows)} turns, {len(u)} user turns -> {out_csv}")

if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
