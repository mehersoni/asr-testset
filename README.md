# Farmer-utterance ASR test sets from helpline call recordings

Scripts for building an ASR test set out of field-test calls to the voice bot: one audio clip per
farmer turn, with a reference transcript, metadata that traces each clip back to its call, and a
NeMo manifest. They were used to build the **Bhili** (Nov 2026, 744 clips) and **Marathi** (Round 2,
Nov 2026, 588 reviewed clips / 48 min) test sets. These are the Marathi versions, generalised so a new
language or field test needs settings, not code changes.

**This repo holds code only.** Recordings, call logs, workbooks, clips, transcripts, `.env` and every
intermediate file stay out of git (see [Data policy](#data-policy)).

## The problem these scripts solve

The call logs give every turn's text (the production ASR for the farmer, the bot's TTS text) and a
timestamp. **The timestamps can't be used to cut audio**: the recordings aren't on a linear clock with
the log (the offset drifts by several seconds within a call). A timestamp-only cut was tried and
rejected (only 60% within 1 s; most pairings wrong). So clips are found by **text alignment**:

```
recording ──VAD──▶ speech segments ──ASR──▶ segment text ─┐
                                                          ├─▶ monotonic DP alignment ─▶ farmer spans
call log ──parse──▶ turns (bot + farmer, in order) ───────┘   (bot turns are the anchors)

still missing?  ─▶ sub-crop search (barge-in) ─▶ Sortformer diarization placement ─▶ listening tool
too much bot?   ─▶ diarization trim
then            ─▶ cut clips + annotation sheet ─▶ correct by ear ─▶ export / final set / WER
```

The bot's turns are long and match their TTS audio closely, so they pin the alignment down; farmer
turns fall in the gaps. The ASR used here only **locates** speech; it never becomes the reference.

## Setup

1. **Make a working folder per test set** and put this repo's `scripts/` in it (clone the repo and
   work inside it, or copy `scripts/`). Every script resolves paths from the folder above `scripts/`,
   and `.gitignore` already keeps the data folders out of git.

   ```
   <lang>_testset/
     scripts/                        this repo
     .env                            ASR server + test-set settings (copy .env.example; never commit)
     raw/recordings/<Month D>/*.wav  call recordings, one subfolder per field-test date
     raw/transcripts/<Month D>/*.txt call logs, same layout
     raw/*.xlsx                      optional field-tester workbook / ASR error review
   ```
   File names must contain the session id: `<date>_<time>_<first 8 hex of session id>.wav|.txt`
   (or start with the full session UUID). The 8-hex id (`sid8`) is how recordings and logs are paired.

2. **Python env**: Python 3.11 with `requirements.txt` (on the shared GPU box this is the
   `bhili-asr` conda env). `parse_transcripts.py`, `export_testset.py`, `make_final.py` and
   `make_drive_ids.py` need only the standard library.

3. **Pick the alignment ASR** (`ASR_BACKEND` in `.env`), the best one you have for the language:
   - `server` (default): VoicEra or any server with OpenAI-style `POST /v1/audio/transcriptions`.
     Set `ASR_BASE_URL`, `ASR_API_KEY`, `ASR_MODEL`, `ASR_LANG`. Marathi used indic-nemotron, `mr`.
   - `local`: a NeMo indic-nemotron `.nemo` checkpoint on the GPU, loaded through VoicEra's
     `model-server/stt/indic-nemotron/asr_engine.py` ([COSS-India/VoicEra](https://github.com/COSS-India/VoicEra)).
     Set `VOICERA_ENGINE_DIR`, `LOCAL_ASR_MODEL`, `BHILI_NEMO_PATH`/`INDIC_NEMO_PATH`, `ASR_LANG`.
     Bhili used the Bhili Nemotron SFT with prompt `bhb`.

   **Decode a few clips before committing to a model and prompt.** The Bhili SFT's `mr` prompt returns
   empty text. Switching the Marathi alignment from the Bhili model (`bhb` prompt) to real Marathi
   indic-nemotron raised clean alignments from 449 to 534 and cut unplaced turns from 103 to 45.

4. **Set the test-set settings** in `.env` or [`scripts/config.py`](scripts/config.py):
   `TESTSET_LANG_CODE` (file-id prefix, ISO 639-3: `bhb`, `mar`, …), `TESTSET_LANGUAGE`,
   `FIELD_TEST_YEAR` (folder names have no year), the bot-greeting phrases that mark a second
   bot language, `BOT_NAME_ASR`, `LOW_SHARE`, `ASR_LABEL`.

## Running it

From the test-set folder. Every ASR step caches its results (`work/voicera_cache.jsonl` for the
server backend), so re-runs only decode new crops.

```bash
python scripts/tester_notes.py                                  # optional: field-tester workbook -> work/tester_notes/
python3 scripts/parse_transcripts.py raw/transcripts work/tester_notes work/turns.csv
python scripts/segment_asr.py raw/recordings work/turns.csv work/segments          # VAD + ASR (slow first time)
python scripts/align.py work/turns.csv work/segments work/alignment.csv
python scripts/recover_overlap.py work/turns.csv work/segments work/alignment.csv work/recovered.csv
python scripts/build_testset.py        # first build: diarize.py validates against utterances.csv
python scripts/diarize.py              # GPU, Sortformer
python scripts/no_transcript.py        # GPU; only if some recordings have no call log
python scripts/diar_place.py
python scripts/build_testset.py
python scripts/diar_trim.py            # cut mostly-bot clips down to the farmer's speech
python scripts/build_testset.py
python scripts/manual_fix/prepare.py   # listening queue for turns still unplaced
# ... listening pass (below), then build_testset.py again ...
# ... correction pass by ear on utterances.csv (below) ...
python3 scripts/make_drive_ids.py raw work/drive_listing_*.json   # optional: Drive links in metadata
python3 scripts/export_testset.py      # testset_v0/: clean vs issue clips, metadata, not_included
python3 scripts/make_final.py          # testset_final/: only review_status=done clips; warns about todo rows
python scripts/eval_wer.py             # WER/CER of the configured ASR on reviewed clips
```

> **`build_testset.py` overwrites `utterances.csv`.** Copy out any corrections before re-running it.

`parse_transcripts.py` takes several roots in priority order (a session found in an earlier root is
skipped later). Pass `raw/recordings` too if some logs were stored with the recordings (Bhili had
this), and the folder of any tester-notes JSON.

### What each script does

| Step | Script | Writes | Notes |
|---|---|---|---|
| 0 | `tester_notes.py` | `work/tester_notes/tester_notes.json` | Optional. Per-call workbook sheets → per-turn English translation + observation. Written for the Marathi Round 2 workbook layout; adapt if yours differs. Hints, never transcripts. |
| 1 | `parse_transcripts.py` | `work/turns.csv` | Three log layouts: `[ts] speaker: text`, the same with a `call_id=` header, and numbered lines. Joins tester notes on (session, turn). |
| 2 | `segment_asr.py` | `work/segments/<sid8>.json` | Silero VAD (250 ms min silence, 120 ms pad, 20 s max) + ASR per segment. |
| 3 | `align.py` | `work/alignment.csv` | Monotonic DP of turns × segments scored by character LCS. `ok` ≥ 0.6 match, `check` 0.35–0.6, otherwise `unaligned`. Never guesses. |
| 4 | `recover_overlap.py` | `work/recovered.csv` | For unaligned farmer turns: decodes sliding sub-crops between the neighbouring aligned turns. This recovers speech that overlaps the bot (barge-in). Rejects crops the bot's own text explains as well. |
| 5 | `build_testset.py` | `audio/`, `utterances.csv`, `manifest_draft.jsonl`, `work/clip_hf_asr.csv` | Cuts clips (16 kHz mono PCM16, +0.1 s pad) and merges every placement source. One row per logged farmer turn; excluded rows say why. |
| 6 | `diarize.py` | `work/diar/<sid8>.json` | `nvidia/diar_streaming_sortformer_4spk-v2.1`, high-latency setting. Bot = the speaker overlapping the aligned bot turns most. Prints validation. |
| 6b | `no_transcript.py` | `work/no_transcript_clips.csv` | Recordings without a log: diarize, bot = most talk time, farmer stretches decoded as the draft text. |
| 7 | `diar_place.py` | `work/diar_located.csv` | Places still-unaligned turns on unclaimed farmer-labelled speech, in conversation order. |
| 8 | `diar_trim.py` | `work/diar_trimmed.csv` | Clips with < 60% farmer speech: trims to the farmer run that best matches the log. |
| 9 | `manual_fix/` | `work/manual/fixes.csv` | Listening tool for what's left (see below). `build_testset.py` applies the decisions. |
| 10 | `apply_error_review.py` | `utterances.csv` | Optional, Marathi-specific policy: apply a text-only wrong→right word review. Fill in `HOLD` / `TESTER_OVERRIDES` for your set. |
| 11 | `export_testset.py` | `testset_v0/` | Clean clips vs `issues/` (`overlap_with_bot`, `possible_bot_speech`, `partial_utterance`, `low_farmer_speech`), `metadata.csv` traced to recording + time range, `not_included.csv`. |
| 12 | `make_final.py` | `testset_final/` | The deliverable: `include=yes, review_status=done` clips only, text NFC-normalised; `audio/`, `transcript/`, `metadata.csv`, `manifest.jsonl`, files named by `file_id`. Reports review status and warns about unreviewed rows. |
| 13 | `eval_wer.py` | `work/eval/asr_hyp.csv` | Offline WER/CER, clean and issue clips separately; `--drafts` for unreviewed rows (agreement, not WER). |
| – | `asr.py`, `voicera_asr.py`, `local_nemo_asr.py`, `config.py` | | ASR backends and settings. |

`file_id` = `<lang>_<YYYYMMDD>_<sid8>_t<turn>` for logged turns and `…_s<n>` for utterances from
recordings with no log. `utt_id` = `<sid8>_t<turn>` (`<sid8>_f<n>`) is the working-sheet id.

## Listening tool (locating the turns automation missed)

`scripts/manual_fix/prepare.py` cuts, for each unplaced turn, the stretch between the neighbouring
aligned turns. It pre-selects the best automatic guess and adds the diarized farmer/bot spans. Then run
`python3 scripts/manual_fix/server.py` and open `http://localhost:8765` (VS Code Remote forwards the
port; otherwise forward it yourself, e.g. `ssh -L 8765:localhost:8765`).

Decisions: *farmer speech = selection*, *not in recording*, *bot only / bot echo*, *unsure*. They are
saved to `work/manual/fixes.csv`; re-run `build_testset.py` to apply them.

## Writing the references (annotators)

`ground_truth` in `utterances.csv` starts as the production ASR. It becomes a reference only after
someone **listens to the clip** and corrects it.

- **Write what is spoken in the clip, not what the log says.** The clip may hold only part of the
  logged turn (`align_status=check`, trimmed clips; see `locator_notes`). Delete words that aren't
  in the clip and add words that are.
- Transcribe the **farmer only**; ignore bot speech in the clip.
- `tester_translation_en` / `tester_observation` describe intent; they are not transcripts.
- Set `review_status` to `done` or `drop` (no usable farmer speech, unintelligible, wrong speaker),
  and fill in `annotator` and `annotator_notes`.

**Settle the writing conventions before annotation starts**, so spelling variation doesn't show up as
WER: script (Devanagari only), one agreed spelling for frequent dialect words (keep a shared word list),
English words in Devanagari as heard (`थँक्यू`, `हॅलो`)?, numbers as words?, no punctuation, fillers
only when they're the whole utterance, cut-off words as heard, `<unk>` for an unintelligible word.

**Keep references unbiased.** Pre-filling from production ASR anchors annotators toward that model.
That's why the alignment ASR's clip text goes to `work/clip_hf_asr.csv` and never into the sheet.
Listening, not proofreading, keeps references fair. Have a second person re-check about 10% of rows
to measure disagreement. Freeze `testset_final/` once it is released; any later fix is a new version.

## Scoring caveats

- **Only listened-to references count.** If the model under test also produced the production ASR
  in the logs (true for Marathi: VoicEra indic-nemotron), unreviewed drafts are its own output.
- **Selection bias.** Clips were located by matching ASR text to the log, so the turns the ASR decoded
  worst are more likely to be among the excluded. Report the excluded count with any WER.
- **Offline vs streaming.** `eval_wer.py` sends whole clips; production streamed the live call.
- **Leakage.** Confirm none of the calls were used to train or fine-tune the model being scored.
- **Bot language.** Some Bhili calls opened with a Marathi bot greeting (`bot_language=marathi`), so
  those farmers may have answered in Marathi. Score them separately, or decide whether they belong.

## Things learned the hard way

- **The farmer's speech is often simply not in the recording** (near-silence where the log shows a
  turn). Expect a sizeable `unaligned` share: Bhili 201 of 855 logged turns had no clip (23 of them in calls
  with no recording), Marathi 45 of 650.
- **Overlap is a diarization problem, not an ASR one.** Farmers speak over the bot on mono
  recordings. Decoding only the farmer's part of an overlapped segment works fine. Sortformer separated bot vs
  farmer well (Bhili: 94.6% of confident farmer clips mostly covered by farmer speech, 98.3% of bot
  segments by bot), but sometimes labels bot phrases as farmer. Always text-match before accepting.
- **Bot echo / repeated words.** Farmers repeat the bot's crop names, and some logged "farmer" turns
  are the bot's own voice. A crop is rejected if the bot's text explains its ASR as well as the farmer's.
- **Barge-in makes long mostly-bot clips** (Marathi: 20–47 s of bot around a 1 s "हो"). That's why
  `diar_trim.py` exists and why Marathi used `LOW_SHARE=0.3` (Bhili 0.1).
- **Text-only bot detection fails** for recordings with no log: short farmer phrases score as high as
  scripted bot lines. Talk time picks the bot (132 of 133 calls agreed with alignment).
- **The VoicEra server garbled 2 of 16 outputs under 8 concurrent requests**; sequential requests were
  deterministic. Keep `ASR_WORKERS=1` unless you've checked your server.
- Tester workbooks can be joined on turn number, but check the join (Marathi: 1271/1274 matched on
  number, speaker and text). A text-only error review was never checked against audio.

## Data policy

Never commit or share call data: recordings, call logs, workbooks, clips, transcripts, `utterances.csv`,
manifests, `work/`, `testset_*/`, Drive listings/ids, ASR caches, or `.env` (server URL and key).
`.gitignore` blocks these by default (all `*.csv`, `*.jsonl`, `*.wav`, `*.xlsx` …). Check
`git status` before every commit anyway. Code comments and examples must not quote farmer speech or real
session ids; use made-up ones.
