"""Farmer utterances from recordings that have no transcript (call log missing or never exported).

With no call log to align against:
  1. diarize with Sortformer (same model and settings as diarize.py)
  2. decode every diarized segment with the ASR backend; the bot is the speaker with the most
     talk time (checked on the 133 logged Bhili calls: agrees with the alignment in 132)
  3. farmer utterance = consecutive farmer segments joined across gaps < JOIN_GAP s with no bot
     segment starting in between; stretches < MIN_DUR s are dropped
  4. each utterance is decoded with the ASR backend; that text is the draft ground truth.
     Nothing is dropped as "bot-like": a text check misses garbled bot speech and also fires on
     6% of confirmed farmer clips, so it only sets review_flag (as does a call whose
     diarization looks unreliable); annotators drop bot clips while listening.
     (transcript_source=config.ASR_LABEL in utterances.csv). Utterances with empty ASR are dropped.

Writes work/diar/<sid8>.json and work/no_transcript_clips.csv. Run in the bhili-asr env:
  CUDA_VISIBLE_DEVICES=0 python scripts/no_transcript.py
"""
import csv, glob, json, os, re, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from align import norm  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402
from recover_overlap import decode_batch  # noqa: E402  (ASR backend: scripts/asr.py)
import config  # noqa: E402
import librosa, soundfile as sf  # noqa: E402

SR, JOIN_GAP, MIN_DUR, PAD, MAX_DUR = 16000, 1.0, 0.3, 0.1, 20.0
MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def sid8_of(name):
    m = re.match(r"([0-9a-f]{8})-", name)
    return m.group(1) if m else re.findall(r"[0-9a-f]{8}", name)[-1]


def main():
    os.chdir(ROOT)
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(MODEL).eval()
    sm = m.sortformer_modules
    sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period, sm.spkcache_len = 340, 40, 40, 300, 188
    sm._check_streaming_parameters()

    logged = {r["sid8"] for r in csv.DictReader(open("work/turns.csv", encoding="utf-8"))}
    bot_lines = sorted({norm(r["text"]) for r in csv.DictReader(open("work/turns.csv", encoding="utf-8"))
                        if r["speaker"] == "assistant" and len(norm(r["text"])) >= 10})
    wavs = [w for w in sorted(glob.glob("raw/recordings/**/*.wav", recursive=True)) if sid8_of(os.path.basename(w)) not in logged]
    os.makedirs("work/diar/tmp", exist_ok=True)
    rows = []
    for w in wavs:
        sid = sid8_of(os.path.basename(w))
        y = librosa.load(w, sr=SR, mono=True)[0]
        dur = len(y) / SR
        tmp = f"work/diar/tmp/{sid}.wav"
        sf.write(tmp, y, SR)
        raw = m.diarize(audio=[tmp], batch_size=1)[0]
        os.remove(tmp)
        dsegs = []
        for line in raw:
            s, e, spk = line.split() if isinstance(line, str) else line
            dsegs.append({"s": float(s), "e": float(e), "spk": spk})
        dsegs.sort(key=lambda x: x["s"])
        if not dsegs:
            print(f"{sid}: no speech found ({dur:.1f}s)", flush=True)
            continue
        texts = decode_batch([y[int(x["s"] * SR):int(x["e"] * SR)] for x in dsegs])
        for x, t in zip(dsegs, texts):
            x["asr"] = t
        # bot = the speaker with the most talk time. On the 133 calls with logs this matches the
        # alignment-based bot speaker in 132. (Matching against scripted bot lines does NOT work:
        # short farmer phrases score as high as bot speech.) Cross-check: who says the bot's name.
        talk = collections.Counter()
        for x in dsegs:
            talk[x["spk"]] += x["e"] - x["s"]
        bot = talk.most_common(1)[0][0]
        name_spk = collections.Counter(x["spk"] for x in dsegs if config.BOT_NAME_ASR in x["asr"]).most_common(1)
        name_spk = name_spk[0][0] if name_spk else None
        if len(talk) == 1:      # one voice only: the bot (e.g. dropped calls); no farmer speech
            bot = next(iter(talk))
        frac = {k: round(v, 1) for k, v in talk.items()}
        for x in dsegs:
            x["role"] = "bot" if x["spk"] == bot else "farmer"
        bot_text = " ".join(x["asr"] for x in dsegs if x["role"] == "bot")
        lang = config.bot_language(bot_text)
        json.dump({"audio": w, "bot_speaker": bot, "n_speakers": len({x["spk"] for x in dsegs}),
                   "talk_time_s": frac, "says_bot_name": name_spk, "segments": dsegs},
                  open(f"work/diar/{sid}.json", "w"), ensure_ascii=False, indent=0)
        # farmer utterances
        spans = []
        for x in dsegs:
            if x["role"] != "farmer":
                continue
            bot_between = spans and any(b["role"] == "bot" and spans[-1][1] <= b["s"] < x["s"] for b in dsegs)
            if spans and x["s"] - spans[-1][1] < JOIN_GAP and not bot_between and x["e"] - spans[-1][0] <= MAX_DUR:
                spans[-1][1] = max(spans[-1][1], x["e"])
            else:
                spans.append([x["s"], x["e"]])
        spans = [(max(0, s - PAD), min(dur, e + PAD)) for s, e in spans if e - s >= MIN_DUR]
        utts = decode_batch([y[int(s * SR):int(e * SR)] for s, e in spans]) if spans else []
        bots = [(x["s"], x["e"]) for x in dsegs if x["role"] == "bot"]
        # diarization looks unreliable when a third voice appears and the non-bot talk time is
        # more than half the bot's (the bot was probably split into two voices; seen in one Bhili call)
        others = sorted((v for k2, v in talk.items() if k2 != bot), reverse=True)
        shaky = len(talk) >= 3 and sum(others) > 0.5 * talk[bot]
        k = 0
        for (s, e), t in zip(spans, utts):
            if not norm(t):
                continue
            k += 1
            over = sum(max(0, min(e, b1) - max(s, b0)) for b0, b1 in bots)
            yv = norm(t)
            bot_match = process.extractOne(yv, bot_lines, scorer=fuzz.partial_ratio)[1] if len(yv) >= 10 else 0
            flags = []
            if bot_match >= 85:
                flags.append("text matches a scripted bot line; may be bot speech")
            if shaky:
                flags.append("diarization unreliable in this call (bot may be split into two voices)")
            rows.append({"utt_id": f"{sid}_f{k:03d}", "sid8": sid, "call_folder": os.path.basename(os.path.dirname(w)),
                         "source_recording": os.path.relpath(w, "raw"), "start": round(s, 3), "end": round(e, 3),
                         "asr_text": t, "overlaps_bot": "yes" if over > 0.3 else "", "bot_language": lang,
                         "n_speakers": len(frac), "bot_name_speaker_agrees": "" if name_spk is None else ("yes" if name_spk == bot else "no"),
                         "review_flag": "; ".join(flags)})
        print(f"{sid} ({os.path.basename(os.path.dirname(w))}, {dur:.0f}s): talk={frac} bot={bot} "
              f"says_name={name_spk} lang={lang} -> {k} farmer utterances", flush=True)
    with open("work/no_transcript_clips.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=['utt_id', 'sid8', 'call_folder', 'source_recording', 'start', 'end', 'asr_text', 'overlaps_bot', 'bot_language', 'n_speakers', 'bot_name_speaker_agrees', 'review_flag']); w.writeheader(); w.writerows(rows)  # header even if empty
    print(f"TOTAL {len(rows)} utterances, {sum(r['end'] - r['start'] for r in rows) / 60:.1f} min, "
          f"over bot speech: {sum(bool(r['overlaps_bot']) for r in rows)}")


if __name__ == "__main__":
    main()
