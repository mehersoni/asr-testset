"""Step 3: cut farmer-utterance clips and write the utterance table for correction.

Inputs : work/turns.csv (parse_transcripts.py), work/alignment.csv (align.py),
         work/recovered.csv (recover_overlap.py; optional) -- farmer turns found by sub-crop
         search, mostly speech overlapping the bot. They become align_status=overlap_recovered
         (or crop_recovered when the crop doesn't touch bot audio).
         work/diar_located.csv (diar_place.py; optional) -- farmer turns placed on farmer speech
         found by Sortformer diarization -> align_status=diar_located.
         work/no_transcript_clips.csv (no_transcript.py; optional) -- farmer utterances from
         recordings with no transcript, found by diarization; ground_truth is pre-filled with the
         alignment ASR (transcript_source=config.ASR_LABEL, align_status=no_transcript_diarized).
         work/manual/fixes.csv (scripts/manual_fix; optional) -- decisions made by listening for
         turns still unaligned: farmer -> align_status=manual_located with the chosen span;
         not_in_recording / bot_only / unsure -> excluded with that reason.
Outputs: audio/<utt_id>.wav            16 kHz mono PCM16 clip of the farmer's speech
         utterances.csv                one row per logged farmer turn (all of them; excluded
                                       rows say why). `ground_truth` is pre-filled with the
                                       production ASR text, to be corrected by listening.
         manifest_draft.jsonl          NeMo-style manifest of the included clips, text = production
                                       ASR (a placeholder until ground truth exists)
         work/clip_hf_asr.csv          alignment ASR of each clip; kept OUT of the annotation
                                       sheet so annotators are not anchored on the model under test

Run in the bhili-asr env: python scripts/build_testset.py
"""
import csv, glob, json, os, re, sys, collections
import librosa, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from align import norm, lcs  # noqa: E402
import config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR, PAD = 16000, 0.10          # extra padding on top of VAD's own 120 ms

EXCLUDE_REASON = {
    "unaligned": "no matching farmer speech found in the recording (absent, too garbled to locate, or too short to match)",
    "no_recording": "no recording for this call",
    "manual_not_in_recording": "checked by listening: the farmer's speech is not in the recording",
    "manual_bot_only": "checked by listening: the logged farmer turn is the bot's own voice",
    "manual_unsure": "checked by listening: could not tell (see locator_notes)",
}
MANUAL_STATUS = {"farmer": "manual_located", "not_in_recording": "manual_not_in_recording",
                 "bot_only": "manual_bot_only", "unsure": "manual_unsure"}


def bot_lang(turns):
    first = next((t["text"] for t in turns if t["speaker"] == "assistant"), "")
    return config.bot_language(first)


def main():
    os.chdir(ROOT)
    turns = collections.defaultdict(list)
    for r in csv.DictReader(open("work/turns.csv", encoding="utf-8")):
        turns[r["sid8"]].append(r)
    lang = {sid: bot_lang(ts) for sid, ts in turns.items()}
    align = list(csv.DictReader(open("work/alignment.csv", encoding="utf-8")))
    rec = {}
    if os.path.exists("work/recovered.csv"):
        rec = {(r["sid8"], r["turn_idx"]): r for r in csv.DictReader(open("work/recovered.csv", encoding="utf-8"))
               if r["accepted"] == "yes"}
    for a in align:
        r = rec.get((a["sid8"], a["turn_idx"]))
        if a["align_status"] == "unaligned" and r:
            a.update(align_status="overlap_recovered" if r["overlaps_bot"] else "crop_recovered",
                     seg_start=r["start"], seg_end=r["end"], seg_asr=r["crop_asr"], match_ratio=r["match_ratio"],
                     _over_bot=r["overlaps_bot"])
    dia = {}
    if os.path.exists("work/diar_located.csv"):
        dia = {(r["sid8"], r["turn_idx"]): r for r in csv.DictReader(open("work/diar_located.csv", encoding="utf-8"))
               if r["accepted"] == "yes"}
    for a in align:
        r = dia.get((a["sid8"], a["turn_idx"]))
        if a["align_status"] == "unaligned" and r:
            a.update(align_status="diar_located", seg_start=r["start"], seg_end=r["end"], seg_asr=r["region_asr"],
                     match_ratio=r["match_ratio"], _over_bot=r["overlaps_bot"])
    trim = {}       # work/diar_trimmed.csv (diar_trim.py): mostly-bot clips cut down to the farmer's speech
    if os.path.exists("work/diar_trimmed.csv"):
        trim = {(r["sid8"], r["turn_idx"]): r for r in csv.DictReader(open("work/diar_trimmed.csv", encoding="utf-8"))
                if r["accepted"] == "yes"}
    for a in align:
        r = trim.get((a["sid8"], a["turn_idx"]))
        if r and a["align_status"] not in ("unaligned", "no_recording"):
            a.update(seg_start=r["start"], seg_end=r["end"], seg_asr=r["trim_asr"], match_ratio=r["match_ratio"],
                     _note=f"trimmed to farmer speech by diarization (was {float(r['orig_start']):.1f}-{float(r['orig_end']):.1f} s)")
    fixes = {}
    if os.path.exists("work/manual/fixes.csv"):
        fixes = {r["utt_id"]: r for r in csv.DictReader(open("work/manual/fixes.csv", encoding="utf-8"))}
    for a in align:
        f = fixes.get(f"{a['sid8']}_t{int(a['turn_idx']):03d}")
        if a["align_status"] == "unaligned" and f:
            a.update(align_status=MANUAL_STATUS[f["decision"]], _note=f["note"], _reviewer=f["reviewer"])
            if f["decision"] == "farmer":
                a.update(seg_start=f["start_s"], seg_end=f["end_s"], seg_asr="", match_ratio="")
    os.makedirs("audio", exist_ok=True)

    cache, rows, manifest, hf = {}, [], [], []
    diar = {}       # sid8 -> farmer-labelled spans from Sortformer diarization (work/diar/), if run

    def farmer_share(sid, s, e):
        if sid not in diar:
            p = f"work/diar/{sid}.json"
            diar[sid] = [(x["s"], x["e"]) for x in json.load(open(p))["segments"] if x["role"] == "farmer"] \
                if os.path.exists(p) else None
        if diar[sid] is None:
            return ""
        return round(sum(max(0.0, min(e, b) - max(s, a)) for a, b in diar[sid]) / max(1e-6, e - s), 2)
    for a in align:
        uid = f"{a['sid8']}_t{int(a['turn_idx']):03d}"
        st = a["align_status"]
        include = st in ("ok", "check", "overlap_recovered", "crop_recovered", "diar_located", "manual_located")
        row = {
            "utt_id": uid, "include": "yes" if include else "no",
            "exclude_reason": "" if include else EXCLUDE_REASON.get(st, st),
            "align_status": st, "match_ratio": a["match_ratio"],
            "bot_speech_in_clip": "", "diar_farmer_share": "", "audio_path": "", "duration_s": "",
            "transcript_source": "production_log", "review_flag": "",
            "production_asr": a["text"], "ground_truth": a["text"] if include else "",
            "review_status": "todo" if include else "", "annotator": "", "annotator_notes": "",
            "locator_notes": " / ".join(x for x in (a.get("_note", ""), a.get("_reviewer", "") and f"located by {a['_reviewer']}") if x),
            "tester_translation_en": a["translation_en"], "tester_observation": a["observation"],
            "session_id": a["session_id"], "sid8": a["sid8"], "call_folder": a["folder"],
            "bot_language": lang[a["sid8"]], "turn_idx": a["turn_idx"], "log_ts_utc": a["ts_utc"],
            "source_recording": os.path.relpath(a["audio"], "raw") if a["audio"] else "",
            "rec_start_s": "", "rec_end_s": "", "transcript_file": a["transcript_file"],
        }
        if include:
            if a["audio"] not in cache:
                cache.clear()
                cache[a["audio"]] = librosa.load(a["audio"], sr=SR, mono=True)[0]
            y = cache[a["audio"]]
            s = max(0.0, float(a["seg_start"]) - PAD)
            e = min(len(y) / SR, float(a["seg_end"]) + PAD)
            path = f"audio/{uid}.wav"
            sf.write(path, y[int(s * SR):int(e * SR)], SR, subtype="PCM_16")
            dur = round(e - s, 3)
            # clip ASR that the bot's own text explains much better than the farmer's logged text
            yv, u = norm(a["seg_asr"]), norm(a["text"])
            bots = [norm(t["text"]) for t in turns[a["sid8"]] if t["speaker"] == "assistant"]
            if a.get("_over_bot") or max((lcs(yv, b) for b in bots), default=0) - lcs(yv, u) >= 8:
                row["bot_speech_in_clip"] = "likely"
            row.update(audio_path=path, duration_s=dur, rec_start_s=round(s, 3), rec_end_s=round(e, 3),
                       diar_farmer_share=farmer_share(a["sid8"], s, e))
            manifest.append({"audio_filepath": path, "duration": dur, "text": a["text"], "utt_id": uid,
                             "session_id": a["session_id"], "bot_language": lang[a["sid8"]]})
            hf.append({"utt_id": uid, config.ASR_LABEL: a["seg_asr"]})
        rows.append(row)

    # recordings with no transcript: farmer utterances found by diarization, drafted by the alignment ASR
    if os.path.exists("work/no_transcript_clips.csv"):
        translated = {}     # July 1: English "Translated Transcript" files in the recordings folder
        for f in glob.glob("raw/recordings/**/*Translated Transcript*.txt", recursive=True):
            m = re.search(r"Session ID:\s*([0-9a-f]{8})", open(f, encoding="utf-8").read())
            if m:
                translated[m.group(1)] = os.path.relpath(f, "raw")
        for c in csv.DictReader(open("work/no_transcript_clips.csv", encoding="utf-8")):
            src = os.path.join("raw", c["source_recording"])
            if src not in cache:
                cache.clear()
                cache[src] = librosa.load(src, sr=SR, mono=True)[0]
            y = cache[src]
            s, e = float(c["start"]), float(c["end"])      # already padded by no_transcript.py
            path = f"audio/{c['utt_id']}.wav"
            sf.write(path, y[int(s * SR):int(e * SR)], SR, subtype="PCM_16")
            dur = round(e - s, 3)
            stem = re.sub(r" \(\d+\)$", "", os.path.splitext(os.path.basename(src))[0])
            rows.append({
                "utt_id": c["utt_id"], "include": "yes", "exclude_reason": "", "align_status": "no_transcript_diarized",
                "match_ratio": "", "bot_speech_in_clip": "likely" if c["overlaps_bot"] else "",
                "diar_farmer_share": farmer_share(c["sid8"], s, e), "audio_path": path, "duration_s": dur,
                "transcript_source": config.ASR_LABEL, "review_flag": c["review_flag"],
                "production_asr": "", "ground_truth": c["asr_text"], "review_status": "todo", "annotator": "",
                "annotator_notes": "", "locator_notes": "", "tester_translation_en": "", "tester_observation": "",
                "session_id": stem, "sid8": c["sid8"], "call_folder": c["call_folder"], "bot_language": c["bot_language"],
                "turn_idx": "", "log_ts_utc": "", "source_recording": c["source_recording"],
                "rec_start_s": round(s, 3), "rec_end_s": round(e, 3), "transcript_file": translated.get(c["sid8"], ""),
            })
            manifest.append({"audio_filepath": path, "duration": dur, "text": c["asr_text"], "utt_id": c["utt_id"],
                             "session_id": stem, "bot_language": c["bot_language"]})

    rows.sort(key=lambda r: (r["call_folder"], r["sid8"], r["utt_id"]))
    with open("utterances.csv", "w", newline="", encoding="utf-8-sig") as fh:   # BOM: Excel shows Devanagari
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open("manifest_draft.jsonl", "w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open("work/clip_hf_asr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt_id", config.ASR_LABEL]); w.writeheader(); w.writerows(hf)

    inc = [r for r in rows if r["include"] == "yes"]
    print(dict(collections.Counter(r["align_status"] for r in rows)))
    print(f"{len(rows)} farmer turns, {len(inc)} clips, {sum(float(r['duration_s']) for r in inc)/60:.1f} min audio")


if __name__ == "__main__":
    main()
