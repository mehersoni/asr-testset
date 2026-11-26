"""Optional: turn a field-testing workbook into tester notes for parse_transcripts.py.

Written for the Marathi Round 2 workbook; adapt it if your workbook differs. Expected layout: one sheet
per call, the call's "Session ID: <uuid>" in the first 3 rows of column A, then a conversation table
(# | time | speaker | original text | English translation | observations) whose # is the log's turn
number. (In the Marathi workbook 1271/1274 turns matched the log on number, speaker and text; check
yours the same way before trusting the join.)
Optional ASR error review (sheet "ASR Errors": sheet name | turn | ... | wrong word | likely word |
confidence): its hints are appended to the observation, marked "error review:".

Settings (environment): TESTER_WORKBOOK (default raw/tester_workbook.xlsx), ERROR_REVIEW (default
raw/asr_error_review.xlsx; skipped if missing), SHEET_PREFIX (only sheets whose name starts with it;
default: every sheet that has a Session ID).

Writes work/tester_notes/tester_notes.json in the _data_group layout parse_transcripts.py reads:
  [{"sessionId", "sheet", "turns": [{"num", "translation", "observation"}]}]
Needs pandas + openpyxl. Run: python scripts/tester_notes.py
"""
import collections, json, os, re
import pandas as pd

WB = os.environ.get("TESTER_WORKBOOK", "raw/tester_workbook.xlsx")
REVIEW = os.environ.get("ERROR_REVIEW", "raw/asr_error_review.xlsx")
PREFIX = os.environ.get("SHEET_PREFIX", "")


def s(v):
    return "" if pd.isna(v) else str(v).strip()


def main():
    sheets = pd.read_excel(WB, sheet_name=None, header=None)
    hints = collections.defaultdict(list)
    review = pd.read_excel(REVIEW, sheet_name="ASR Errors").itertuples(index=False) if os.path.exists(REVIEW) else []
    for r in review:
        if s(r[0]) and s(r[1]):
            hints[(s(r[0]), int(float(r[1])))].append(f"{s(r[4])} → {s(r[5])}")
    calls = []
    for name, d in sheets.items():
        if not name.startswith(PREFIX):
            continue
        m = re.search(r"Session ID: ([0-9a-f-]{36})", " ".join(d.iloc[:3, 0].astype(str)))
        if not m:
            print("skip sheet (no Session ID):", name)
            continue
        hdr = next(i for i in range(len(d)) if s(d.iloc[i, 0]) == "#")
        turns = []
        for i in range(hdr + 1, len(d)):
            try:
                n = int(d.iloc[i, 0])
            except (TypeError, ValueError):
                continue
            obs = s(d.iloc[i, 5])
            if hints.get((name, n)):
                obs = (obs + " | " if obs else "") + "error review: " + "; ".join(hints[(name, n)])
            turns.append({"num": n, "translation": s(d.iloc[i, 4]), "observation": obs})
        calls.append({"sessionId": m.group(1), "sheet": name, "turns": turns})
    os.makedirs("work/tester_notes", exist_ok=True)
    json.dump(calls, open("work/tester_notes/tester_notes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(calls)} call sheets, {sum(len(c['turns']) for c in calls)} turns, "
          f"{sum(len(v) for v in hints.values())} error-review hints -> work/tester_notes/tester_notes.json")


if __name__ == "__main__":
    main()
