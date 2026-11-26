"""Export the test set as a self-contained folder (re-run after corrections to refresh it).

  testset_v0/
    audios/<file_id>.wav          clips with no known issue
    transcripts/<file_id>.txt     transcript of each clip in audios/ (one line, UTF-8)
    issues/audios/<file_id>.wav   clips with overlap with the bot or another issue
    issues/transcripts/<file_id>.txt
    issues/issues.csv             one row per issue clip: issue types and details
    metadata.csv                  every clip, traced back to its call recording, time range,
                                  source transcript and Google Drive links
    not_included.csv              logged farmer utterances with no clip, and why
    manifest.jsonl                NeMo-style manifest of all clips (paths relative to the folder)
    README.md

file_id = <lang>_<field-test date>_<call id>_t<turn> for utterances from a call log, or
          <lang>_<field-test date>_<call id>_s<n> for utterances found in recordings with no
          transcript (<lang> = config.LANG_CODE, the ISO 639-3 code: bhb Bhili, mar Marathi).

Transcripts are the current `ground_truth` column of utterances.csv; until the correction pass
is done they are drafts (review_status=todo in metadata.csv).

Run: python3 scripts/export_testset.py [out_dir]     (default testset_v0; the folder is rebuilt)
"""
import csv, json, os, re, shutil, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
OVERLAP_MIN, LOW_SHARE = config.OVERLAP_MIN, config.LOW_SHARE
MARK = ".generated_by_export_testset"


def field_test_date(folder):
    m = re.match(r"([A-Za-z]+)\s+(\d+)", folder)
    return f"{config.FIELD_TEST_YEAR}{MONTHS[m.group(1).lower()]:02d}{int(m.group(2)):02d}"


def drive(ids, path):
    i = ids.get(path)
    return f"https://drive.google.com/file/d/{i}/view" if i else ""


def main(out="testset_v0"):
    os.chdir(ROOT)
    out = os.path.join(ROOT, out)
    if os.path.exists(out):
        if not os.path.exists(os.path.join(out, MARK)):
            sys.exit(f"{out} exists and wasn't made by this script; not overwriting it")
        shutil.rmtree(out)
    for d in ("audios", "transcripts", "issues/audios", "issues/transcripts"):
        os.makedirs(os.path.join(out, d))
    open(os.path.join(out, MARK), "w").close()

    # raw/-relative path -> Google Drive file id, for the *_drive_url columns (scripts/make_drive_ids.py).
    # Optional: without it the URL columns are left empty.
    ids = json.load(open("work/drive_ids.json", encoding="utf-8")) if os.path.exists("work/drive_ids.json") else {}
    utt = list(csv.DictReader(open("utterances.csv", encoding="utf-8-sig")))
    bot = {}

    def bot_spans(sid):
        if sid not in bot:
            p = f"work/diar/{sid}.json"
            bot[sid] = [(x["s"], x["e"]) for x in json.load(open(p))["segments"] if x["role"] == "bot"] \
                if os.path.exists(p) else []
        return bot[sid]

    meta, issues, manifest, excluded = [], [], [], []
    for u in sorted(utt, key=lambda x: (field_test_date(x["call_folder"]), x["sid8"], x["utt_id"])):
        date = field_test_date(u["call_folder"])
        tail = f"t{int(u['turn_idx']):03d}" if u["turn_idx"] else "s" + u["utt_id"].split("_f")[-1]
        fid = f"{config.LANG_CODE}_{date}_{u['sid8']}_{tail}"
        rec_url = drive(ids, u["source_recording"])
        tr_url = drive(ids, u["transcript_file"])
        if u["include"] != "yes":
            excluded.append({"file_id": fid, "utt_id": u["utt_id"], "logged_text": u["production_asr"],
                             "reason": u["exclude_reason"], "status": u["align_status"], "field_test": u["call_folder"],
                             "field_test_date": date, "session_id": u["session_id"], "call_id": u["sid8"],
                             "turn_idx": u["turn_idx"], "log_ts_utc": u["log_ts_utc"],
                             "source_recording": u["source_recording"], "recording_drive_url": rec_url,
                             "source_transcript": u["transcript_file"], "transcript_drive_url": tr_url})
            continue
        s, e = float(u["rec_start_s"]), float(u["rec_end_s"])
        dur = float(u["duration_s"])
        ov = sum(max(0.0, min(e, b) - max(s, a)) for a, b in bot_spans(u["sid8"]))
        kinds, details = [], []
        if ov > OVERLAP_MIN:
            kinds.append("overlap_with_bot")
            details.append(f"{ov:.1f} s of the {dur:.1f} s clip ({ov / dur:.0%}) overlaps bot speech; transcribe the farmer only")
        if u["review_flag"] or (u["bot_speech_in_clip"] == "likely" and ov <= OVERLAP_MIN):
            kinds.append("possible_bot_speech")
            details.append(u["review_flag"] or "the clip's ASR resembles the bot's words; check it is the farmer")
        if u["align_status"] == "check":
            kinds.append("partial_utterance")
            details.append(f"may hold only part of the logged sentence (text match {float(u['match_ratio']):.2f}); "
                           "transcribe only what is spoken")
        if u["diar_farmer_share"] != "" and float(u["diar_farmer_share"]) < LOW_SHARE:
            kinds.append("low_farmer_speech")
            details.append(f"diarization heard the farmer in only {float(u['diar_farmer_share']):.0%} of the clip")
        sub = "issues/" if kinds else ""
        apath, tpath = f"{sub}audios/{fid}.wav", f"{sub}transcripts/{fid}.txt"
        shutil.copyfile(u["audio_path"], os.path.join(out, apath))
        with open(os.path.join(out, tpath), "w", encoding="utf-8") as fh:
            fh.write(u["ground_truth"].strip() + "\n")
        row = {"file_id": fid, "audio_path": apath, "transcript_path": tpath, "duration_s": dur,
               "transcript": u["ground_truth"], "review_status": u["review_status"], "has_issues": "yes" if kinds else "no",
               "issue_types": ";".join(kinds), "field_test": u["call_folder"], "field_test_date": date,
               "session_id": u["session_id"], "call_id": u["sid8"], "turn_idx": u["turn_idx"], "log_ts_utc": u["log_ts_utc"],
               "source_recording": u["source_recording"], "rec_start_s": s, "rec_end_s": e, "recording_drive_url": rec_url,
               "source_transcript": u["transcript_file"], "transcript_drive_url": tr_url,
               "production_asr": u["production_asr"], "transcript_source": u["transcript_source"],
               "bot_language": u["bot_language"], "how_found": u["align_status"], "match_ratio": u["match_ratio"],
               "diar_farmer_share": u["diar_farmer_share"], "bot_overlap_s": round(ov, 2), "tester_translation_en": u["tester_translation_en"],
               "tester_observation": u["tester_observation"], "utt_id": u["utt_id"]}
        meta.append(row)
        manifest.append({"audio_filepath": apath, "duration": dur, "text": u["ground_truth"].strip(),
                         "file_id": fid, "has_issues": bool(kinds)})
        if kinds:
            issues.append({"file_id": fid, "audio_path": apath, "transcript_path": tpath, "duration_s": dur,
                           "issue_types": ";".join(kinds), "issue_details": " | ".join(details),
                           "bot_overlap_s": round(ov, 2), "bot_overlap_share": round(ov / dur, 2),
                           "diar_farmer_share": u["diar_farmer_share"], "how_found": u["align_status"],
                           "transcript": u["ground_truth"], "field_test": u["call_folder"], "call_id": u["sid8"],
                           "turn_idx": u["turn_idx"], "source_recording": u["source_recording"],
                           "rec_start_s": s, "rec_end_s": e, "recording_drive_url": rec_url})

    ISSUE_FIELDS = ["file_id", "audio_path", "transcript_path", "duration_s", "issue_types", "issue_details",
                    "bot_overlap_s", "bot_overlap_share", "diar_farmer_share", "how_found", "transcript", "field_test",
                    "call_id", "turn_idx", "source_recording", "rec_start_s", "rec_end_s", "recording_drive_url"]
    EXCLUDED_FIELDS = ["file_id", "utt_id", "logged_text", "reason", "status", "field_test", "field_test_date",
                       "session_id", "call_id", "turn_idx", "log_ts_utc", "source_recording", "recording_drive_url",
                       "source_transcript", "transcript_drive_url"]

    def write(path, rows, fields=None):
        with open(os.path.join(out, path), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)
    if not meta:
        sys.exit("no included clips in utterances.csv; nothing to export")
    write("metadata.csv", meta)
    write("issues/issues.csv", issues, ISSUE_FIELDS)
    write("not_included.csv", excluded, EXCLUDED_FIELDS)
    with open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    clean = [r for r in meta if r["has_issues"] == "no"]
    kinds = collections.Counter(k for r in issues for k in r["issue_types"].split(";"))
    mins = lambda rows: sum(float(r["duration_s"]) for r in rows) / 60
    reasons = collections.Counter(r["status"] for r in excluded)
    readme = f"""# {config.LANGUAGE} farmer-utterance ASR test set (v0)

Farmer utterances from {config.SOURCE_DESC} ({", ".join(sorted({r["field_test"] for r in meta}))}),
one clip per utterance, 16 kHz mono 16-bit WAV, each with a one-line UTF-8 transcript.

| | Clips | Minutes |
|---|---|---|
| `audios/` + `transcripts/` (no known issue) | {len(clean)} | {mins(clean):.1f} |
| `issues/audios/` + `issues/transcripts/` | {len(issues)} | {mins(issues):.1f} |
| **Total** | **{len(meta)}** | **{mins(meta):.1f}** |

Issue types in `issues/issues.csv` (a clip can have several):
{chr(10).join(f"- `{k}`: {v}" for k, v in kinds.most_common())}

**Transcripts are drafts until the correction pass is done** (`review_status` in `metadata.csv`):
production ASR text from the call log, or the alignment ASR ({config.ASR_LABEL}) for recordings
with no transcript. `tester_translation_en` / `tester_observation` (field-tester notes) are hints
about what the farmer meant, not transcripts.

## Naming
`{config.LANG_CODE}_<field-test date>_<call id>_t<turn>`: the farmer's turn number in that call's log.
`{config.LANG_CODE}_<field-test date>_<call id>_s<n>`: the n-th farmer utterance found in a recording that has no transcript.
The audio and transcript of a clip share the same base name.

## Tracing a clip back
`metadata.csv` gives, for every clip: the field test, session/call id, turn number and log timestamp,
the source call recording and the clip's start/end time in it (`rec_start_s`, `rec_end_s`), the source
transcript file, and Google Drive links to both. `utt_id` is the clip's id in the working sheet
`utterances.csv`.

## Not included
`not_included.csv` lists the {len(excluded)} logged farmer utterances with no clip:
{chr(10).join(f"- {v} × {k}" for k, v in reasons.most_common())}

Regenerate with `python3 scripts/export_testset.py` (rebuilds this folder from `utterances.csv`).
"""
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(readme)
    print(f"{len(meta)} clips ({mins(meta):.1f} min): {len(clean)} clean ({mins(clean):.1f} min), "
          f"{len(issues)} with issues ({mins(issues):.1f} min) {dict(kinds)}; {len(excluded)} not included -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
