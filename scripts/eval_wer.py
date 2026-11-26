"""WER/CER of the configured ASR (scripts/asr.py; ASR_BACKEND, ASR_MODEL, ASR_LANG) on the test set.

If the model under test is the one that produced the production ASR in the call logs (as for the
Marathi set), the draft references (production_asr, plus any text-only fixes) are its own output.
Only rows with review_status=done are scored by default; --drafts also scores todo rows, reported
separately and labelled as not a WER.

Clean clips and clips with issues (bot overlap, partial turn, low farmer speech) are reported
separately: issue clips mix in bot speech or cut words, which shows up as insertions or deletions
that are not the ASR's fault.

Decoding is offline (whole clip, one request at a time; concurrent requests garble some outputs)
through scripts/asr.py; the server backend caches separately in work/eval/voicera_eval_cache.jsonl.
Production decoded the live call in streaming mode, so offline numbers can differ from what callers saw.

Writes work/eval/asr_hyp.csv (per clip: ref, hyp, errors) and prints the summary.
Run in the bhili-asr env: python scripts/eval_wer.py [--drafts]
"""
import collections, csv, os, re, sys, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

PUNCT = re.compile(r"[।॥.,!?;:\"'“”‘’()\[\]{}<>…–—\-/\\|*_~`]+")


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    t = t.replace("‌", "").replace("‍", "")  # ZWNJ/ZWJ
    return " ".join(PUNCT.sub(" ", t).split())


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def main():
    drafts = "--drafts" in sys.argv
    # own cache, sequential requests: the shared cache holds outputs from concurrent requests,
    # which the server sometimes garbles (see voicera_asr.py)
    os.environ.setdefault("ASR_CACHE", os.path.join(ROOT, "work", "eval", "voicera_eval_cache.jsonl"))
    os.environ["ASR_WORKERS"] = "1"
    os.makedirs(os.path.join(ROOT, "work", "eval"), exist_ok=True)
    import asr
    rows = [r for r in csv.DictReader(open("utterances.csv", encoding="utf-8-sig")) if r["include"] == "yes"]
    status = collections.Counter(r["review_status"] or "todo" for r in rows)
    rows = [r for r in rows if r["review_status"] == "done" or (drafts and r["review_status"] == "todo")]
    issues = {r["utt_id"]: r["issue_types"] for r in csv.DictReader(open("testset_v0/metadata.csv", encoding="utf-8-sig"))
              if r["has_issues"] == "yes"}
    print(f"review status: {dict(status)}; scoring {len(rows)} rows")
    if not rows:
        sys.exit("No rows with review_status=done yet; nothing valid to score. (--drafts for a provisional run.)")

    # if audio/ was cleaned up after make_final.py, fall back to testset_final/audio/ (clips renamed to file_id)
    final = {}
    if os.path.exists("testset_final/metadata.csv"):
        final = {m["utt_id"]: os.path.join("testset_final", m["audio_filepath"])
                 for m in csv.DictReader(open("testset_final/metadata.csv", encoding="utf-8-sig"))}
    path = lambda r: r["audio_path"] if os.path.exists(r["audio_path"]) else final[r["utt_id"]]
    hyps = asr.decode_batch([asr.load_16k(path(r)) for r in rows])
    out, agg = [], collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # word err, words, char err, chars, n
    for r, h in zip(rows, hyps):
        ref, hyp = norm(r["ground_truth"]), norm(h)
        we, ce = ed(ref.split(), hyp.split()), ed(ref.replace(" ", ""), hyp.replace(" ", ""))
        grp = ("verified" if r["review_status"] == "done" else "DRAFT (model's own text)") + \
              (" / issues" if r["utt_id"] in issues else " / clean")
        for g in (grp, grp.split(" / ")[0] + " / all"):
            a = agg[g]
            a[0] += we; a[1] += len(ref.split()); a[2] += ce; a[3] += len(ref.replace(" ", "")); a[4] += 1
        out.append({"utt_id": r["utt_id"], "group": grp, "issue_types": issues.get(r["utt_id"], ""),
                    "ref": ref, "hyp": hyp, "word_errors": we, "ref_words": len(ref.split())})

    os.makedirs("work/eval", exist_ok=True)
    with open("work/eval/asr_hyp.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{'group':38} {'clips':>5} {'words':>6} {'WER':>7} {'CER':>7}")
    for g in sorted(agg):
        we, nw, ce, nc, n = agg[g]
        print(f"{g:38} {n:5} {nw:6} {we / max(nw, 1):7.1%} {ce / max(nc, 1):7.1%}")
    if drafts:
        print("DRAFT rows are scored against the model's own (streaming) output: that is agreement, not WER.")
    print("per-clip -> work/eval/asr_hyp.csv")


if __name__ == "__main__":
    main()
