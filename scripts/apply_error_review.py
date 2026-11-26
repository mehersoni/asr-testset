"""Optional: apply a text-only ASR error review to `ground_truth` in utterances.csv.

Used for the Marathi set, where the team's reference policy was "the logged text is correct unless the
error review fixed it". Use it only if your test set has the same policy; otherwise references come
from annotators listening to the clips (see README).

Sources:
- ERROR_REVIEW (default raw/asr_error_review.xlsx), sheet "ASR Errors": one row per wrong word/phrase
  (sheet name, turn, ..., wrong, likely correct, confidence). Text-only review: audio was NOT checked.
- TESTER_OVERRIDES below: corrections stated by field testers who heard the call. They replace the
  error review's edits for that turn.
- Sheet "Garbled Utterances": no correction possible; noted on the row for the audio pass.
The sheet name -> session id map comes from work/tester_notes/tester_notes.json (tester_notes.py).

ground_truth is rebuilt from production_asr each run, so the script is idempotent and can be re-run
after build_testset.py (which overwrites utterances.csv). Rows that already have review_status=done
or drop (audio-checked by an annotator) are left alone.

Reference policy (Marathi set, Nov 2026): the logged text is correct unless the error review fixed it, so
every included row becomes review_status=done with annotator=log+error_review (edits and confidence
in annotator_notes), except clips the review calls garbled and clips in HOLD, which stay todo.
Rows a person set to done/drop (another annotator) are never touched.
Replacements match whole words only, so "स" -> "पाऊस" does not touch words that contain स.

Run: python3 scripts/apply_error_review.py   (needs pandas + openpyxl)
"""
import collections, csv, json, re, sys
import pandas as pd

import os

REVIEW = os.environ.get("ERROR_REVIEW", "raw/asr_error_review.xlsx")
CSV = "utterances.csv"
NOTES = "work/tester_notes/tester_notes.json"
LOG = "work/error_review_applied.csv"

ANNOTATOR = "log+error_review"
# Per-test-set decisions: fill these in for your own set (the Marathi entries are not shipped here).
# utt_id -> note. Clips whose audio doesn't hold the logged text; kept todo until the text is fixed by ear.
HOLD = {
    # "abcd1234_t002": "clip holds only part of the logged turn, text needs fixing by ear",
}

# (sheet, turn) -> list of (wrong, correct), replacing the error review's edits for that turn.
# Use when a tester who heard the call says what was actually said.
TESTER_OVERRIDES = {
    # ("<sheet name>", 9): [("<wrong word>", "<word the tester heard>")],
}


def s(v):
    return "" if pd.isna(v) else str(v).strip()


def replace_words(text, wrong, right):
    """Replace whole-word occurrences of `wrong`; return (new_text, count)."""
    pat = r"(?<!\S)" + r"\s+".join(map(re.escape, wrong.split())) + r"(?!\S)"
    return re.subn(pat, right, text)


def main():
    sheet_sid = {c["sheet"]: c["sessionId"] for c in json.load(open(NOTES, encoding="utf-8"))}
    edits = collections.defaultdict(list)  # (sheet, turn) -> [(wrong, right, confidence)]
    for r in pd.read_excel(REVIEW, sheet_name="ASR Errors").itertuples(index=False):
        if s(r[0]) and s(r[4]):
            edits[(s(r[0]), int(float(r[1])))].append((s(r[4]), s(r[5]), s(r[6])))
    for k, v in TESTER_OVERRIDES.items():
        edits[k] = [(w, c, "Tester") for w, c in v]
    garbled = {(s(r[0]), int(float(r[1]))): s(r[3])
               for r in pd.read_excel(REVIEW, sheet_name="Garbled Utterances").itertuples(index=False) if s(r[0])}

    with open(CSV, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        fields, rows = rd.fieldnames, list(rd)
    by_key = {(r["session_id"], r["turn_idx"]): r for r in rows}

    log, st, handled = [], collections.Counter(), set()
    for key in sorted(set(edits) | set(garbled)):
        sheet, turn = key
        r = by_key.get((sheet_sid.get(sheet), str(turn)))
        if r is None:
            st["no_row"] += 1
            print("no utterance row for", key, file=sys.stderr)
            continue
        if (r["review_status"] in ("done", "drop") and r["annotator"] != ANNOTATOR) or r["include"] != "yes":
            st["skipped_" + (r["review_status"] or "excluded")] += 1
            continue
        text, notes = r["production_asr"], []
        for wrong, right, conf in edits.get(key, []):
            text, n = replace_words(text, wrong, right)
            if n == 0 and text.count(wrong) == 1:  # wrong word is part of a longer token
                text, n = text.replace(wrong, right), 1
            st["applied" if n else "not_found"] += 1
            log.append({"utt_id": r["utt_id"], "sheet": sheet, "turn": turn, "wrong": wrong, "correct": right,
                        "confidence": conf, "replaced": n})
            if n:
                notes.append(f"{wrong} → {right} ({conf})")
        if key in garbled:
            notes.append("garbled per error review, needs audio: " + garbled[key])
            st["garbled_noted"] += 1
        r["ground_truth"] = text
        r["annotator"] = ANNOTATOR
        r["annotator_notes"] = "; ".join(notes)
        r["review_status"] = "todo" if key in garbled else "done"
        handled.add(r["utt_id"])

    # Reference policy (user, 2026-11-26): the log text is the reference unless the error review fixed
    # it. Every other included row is final as logged, except rows whose clip doesn't hold the logged text.
    for r in rows:
        if r["include"] != "yes" or r["utt_id"] in handled:
            continue
        if r["review_status"] in ("done", "drop") and r["annotator"] != ANNOTATOR:
            continue  # decided by a person
        r["ground_truth"], r["annotator"] = r["production_asr"], ANNOTATOR
        if r["utt_id"] in HOLD:
            r["review_status"], r["annotator_notes"] = "todo", HOLD[r["utt_id"]]
        else:
            r["review_status"], r["annotator_notes"] = "done", "log text as is"

    with open(CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)
    with open(LOG, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, ["utt_id", "sheet", "turn", "wrong", "correct", "confidence", "replaced"])
        w.writeheader()
        w.writerows(log)
    print(dict(st))
    print(collections.Counter(r["review_status"] for r in rows if r["include"] == "yes"))
    print("edit log ->", LOG)


if __name__ == "__main__":
    main()
