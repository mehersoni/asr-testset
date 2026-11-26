"""Step 2d: speaker diarization of every call with NVIDIA Streaming Sortformer
(nvidia/diar_streaming_sortformer_4spk-v2.1), then label which diarized speaker is the bot.

Why: the recordings are mono, and the farmer often speaks over the bot. VAD then puts both in
one segment and the alignment labels the whole stretch as bot. Sortformer marks each
speaker's time ranges directly from the audio, overlaps included, so the farmer's speech
inside "bot" stretches becomes visible.

Bot identification: per call, the diarized speaker with the most overlap with segments that
align.py assigned to bot turns. Every other speaker counts as farmer (a call can have more
than one human voice).

Validation printed at the end (against what is already known):
  - farmer clips with align_status=ok: share of each clip covered by farmer-labelled speech
  - segments aligned to bot turns: share covered by the bot speaker

Writes work/diar/<sid8>.json = {"audio", "bot_speaker", "segments": [{"s","e","spk","role"}]}

Run in the bhili-asr env:
  CUDA_VISIBLE_DEVICES=0 python scripts/diarize.py
"""
import csv, json, os, sys, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from align import align_call  # noqa: E402

SR = 16000
MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def covered(s, e, spans):
    return sum(overlap(s, e, a, b) for a, b in spans) / max(1e-6, e - s)


def main():
    os.chdir(ROOT)
    import librosa, soundfile as sf
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(MODEL).eval()
    sm = m.sortformer_modules     # model card's high-latency streaming setting (best accuracy)
    sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period, sm.spkcache_len = 340, 40, 40, 300, 188
    sm._check_streaming_parameters()

    turns = collections.defaultdict(list)
    for r in csv.DictReader(open("work/turns.csv", encoding="utf-8")):
        turns[r["sid8"]].append(r)
    utt = collections.defaultdict(list)
    for u in csv.DictReader(open("utterances.csv", encoding="utf-8-sig")):
        utt[u["sid8"]].append(u)
    os.makedirs("work/diar/tmp", exist_ok=True)
    val_f, val_b = [], []
    for sid in sorted(turns):
        f = f"work/segments/{sid}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        out = f"work/diar/{sid}.json"
        if not os.path.exists(out):
            y = librosa.load(d["audio"], sr=SR, mono=True)[0]
            tmp = f"work/diar/tmp/{sid}.wav"
            sf.write(tmp, y, SR)
            raw = m.diarize(audio=[tmp], batch_size=1)[0]
            os.remove(tmp)
            dsegs = []
            for line in raw:
                s, e, spk = line.split() if isinstance(line, str) else line
                dsegs.append({"s": float(s), "e": float(e), "spk": spk})
            # which speaker is the bot: most overlap with segments aligned to bot turns
            segs, ts = d["segments"], turns[sid]
            assign = align_call(ts, segs)
            bot_spans = [(segs[k]["s"], segs[k]["e"]) for i, (a, b) in assign.items()
                         if ts[i]["speaker"] == "assistant" for k in range(a, b)]
            score = collections.Counter()
            for x in dsegs:
                score[x["spk"]] += sum(overlap(x["s"], x["e"], a, b) for a, b in bot_spans)
            bot = score.most_common(1)[0][0] if score else None
            for x in dsegs:
                x["role"] = "bot" if x["spk"] == bot else "farmer"
            json.dump({"audio": d["audio"], "bot_speaker": bot, "n_speakers": len({x["spk"] for x in dsegs}),
                       "segments": sorted(dsegs, key=lambda x: x["s"])}, open(out, "w"), indent=0)
        dj = json.load(open(out))
        far = [(x["s"], x["e"]) for x in dj["segments"] if x["role"] == "farmer"]
        botd = [(x["s"], x["e"]) for x in dj["segments"] if x["role"] == "bot"]
        for u in utt[sid]:
            if u["align_status"] == "ok" and u["rec_start_s"]:
                val_f.append(covered(float(u["rec_start_s"]), float(u["rec_end_s"]), far))
        segs, ts = d["segments"], turns[sid]
        for i, (a, b) in align_call(ts, segs).items():
            if ts[i]["speaker"] == "assistant":
                for k in range(a, b):
                    if segs[k]["e"] - segs[k]["s"] >= 1.0:
                        val_b.append(covered(segs[k]["s"], segs[k]["e"], botd))
        print(f"{sid}: {dj['n_speakers']} speakers, bot={dj['bot_speaker']}, "
              f"farmer {sum(e - s for s, e in far):.1f}s", flush=True)
    vf, vb = np.array(val_f), np.array(val_b)
    print(f"VALIDATION farmer clips (ok): n={len(vf)} median coverage by farmer speech {np.median(vf):.0%}, "
          f">=50% covered: {(vf >= .5).mean():.1%}, <10% covered: {(vf < .1).mean():.1%}")
    print(f"VALIDATION bot segments: n={len(vb)} median coverage by bot speech {np.median(vb):.0%}, "
          f">=50% covered: {(vb >= .5).mean():.1%}")


if __name__ == "__main__":
    main()
