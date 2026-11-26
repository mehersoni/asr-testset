"""Build testset_final/: the finished test set (review_status=done rows only).

  testset_final/audio/<file_id>.wav        16 kHz mono PCM16 clips
  testset_final/transcript/<file_id>.txt   the clip's transcription (UTF-8, one line)
  testset_final/metadata.csv               audio_filepath, transcript_filepath, transcription, then all metadata
  testset_final/manifest.jsonl             NeMo manifest: audio_filepath, duration, text, then the same metadata

Reads utterances.csv (references) and testset_v0/metadata.csv (export metadata, clip files).
Rows marked review_status=drop are left out. Included rows still todo are reported, so a
half-reviewed sheet can't silently become a test set.

Run after export_testset.py: python3 scripts/make_final.py
"""
import collections, csv, json, os, shutil, unicodedata, wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "testset_final"


def clean(t):
    return " ".join(unicodedata.normalize("NFC", t or "").split())


def main():
    utt = {r["utt_id"]: r for r in csv.DictReader(open("utterances.csv", encoding="utf-8-sig"))}
    v0 = {r["utt_id"]: r for r in csv.DictReader(open("testset_v0/metadata.csv", encoding="utf-8-sig"))}
    if os.path.exists(OUT):
        assert set(os.listdir(OUT)) <= {"audio", "transcript", "metadata.csv", "manifest.jsonl"}, f"{OUT} holds other files"
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/audio")
    os.makedirs(f"{OUT}/transcript")

    rows = []
    for uid, u in sorted(utt.items()):
        if u["include"] != "yes" or u["review_status"] != "done":
            continue
        m = v0[uid]
        text = clean(u["ground_truth"])
        assert text, uid
        src = os.path.join("testset_v0", m["audio_path"])
        dst = f"audio/{m['file_id']}.wav"
        tdst = f"transcript/{m['file_id']}.txt"
        with open(os.path.join(OUT, tdst), "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        shutil.copy2(src, os.path.join(OUT, dst))
        with wave.open(os.path.join(OUT, dst)) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1, uid
            dur = round(w.getnframes() / 16000, 3)
        edited = clean(u["production_asr"]) != text
        rows.append({
            # core
            "audio_filepath": dst,
            "transcript_filepath": tdst,
            "transcription": text,
            # clip
            "file_id": m["file_id"], "utt_id": uid, "duration_s": dur,
            "has_issues": m["has_issues"], "issue_types": m["issue_types"],
            # transcript provenance
            "original_asr": clean(u["production_asr"]),
            "transcription_source": "error_review_correction" if edited else "call_log_asr",
            "transcription_edits": u["annotator_notes"] if edited else "",
            # call
            "field_test": m["field_test"], "field_test_date": m["field_test_date"],
            "session_id": m["session_id"], "call_id": m["call_id"], "turn_idx": m["turn_idx"],
            "log_ts_utc": m["log_ts_utc"], "bot_language": m["bot_language"],
            # source and location in the recording
            "source_recording": m["source_recording"], "rec_start_s": m["rec_start_s"], "rec_end_s": m["rec_end_s"],
            "recording_drive_url": m["recording_drive_url"], "source_transcript": m["source_transcript"],
            "transcript_drive_url": m["transcript_drive_url"],
            "how_found": m["how_found"], "match_ratio": m["match_ratio"],
            "diar_farmer_share": m["diar_farmer_share"], "bot_overlap_s": m["bot_overlap_s"],
            "locator_notes": u["locator_notes"],
            # field tester notes (tester_notes.py)
            "tester_translation_en": m["tester_translation_en"], "tester_observation": m["tester_observation"],
        })

    if not rows:
        raise SystemExit("no rows with include=yes and review_status=done; nothing to put in the final set")
    with open(f"{OUT}/metadata.csv", "w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel shows Devanagari
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            j = {"audio_filepath": r["audio_filepath"], "duration": r["duration_s"], "text": r["transcription"]}
            j.update({k: v for k, v in r.items() if k not in ("audio_filepath", "duration_s", "transcription")})
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    iss = sum(r["has_issues"] == "yes" for r in rows)
    st = collections.Counter(u["review_status"] or "todo" for u in utt.values() if u["include"] == "yes")
    print(f"review status: {dict(st)}")
    if st.get("todo"):
        print(f"WARNING: {st['todo']} included rows are not reviewed yet and are left out")
    print(f"{len(rows)} clips, {sum(r['duration_s'] for r in rows) / 60:.1f} min "
          f"({len(rows) - iss} clean, {iss} with issues), "
          f"{sum(r['transcription_source'] == 'error_review_correction' for r in rows)} corrected -> {OUT}/")


if __name__ == "__main__":
    main()
