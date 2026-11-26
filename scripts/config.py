"""Per-test-set settings. Change them here or override with environment variables / the .env file.

Values that came from the Bhili and Marathi sets are noted, so a new language can start from
the closer one. ASR settings live in the environment / .env (see scripts/asr.py).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.environ.get("ASR_ENV_FILE") or os.path.join(ROOT, ".env")


def load_env(path=ENV_FILE):
    """KEY=VALUE lines from the .env file into os.environ; real environment variables win.
    Blank lines, # comments and trailing ' # comments' are ignored."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

# ISO 639-3 code used as the file_id prefix in the export: <code>_<YYYYMMDD>_<call id>_t<turn>
LANG_CODE = os.environ.get("TESTSET_LANG_CODE", "mar")            # Bhili: bhb, Marathi: mar
# Human-readable name, used in the exported README and as the default bot_language label
LANGUAGE = os.environ.get("TESTSET_LANGUAGE", "Marathi")
# Short description of where the calls come from, for the exported README
SOURCE_DESC = os.environ.get("TESTSET_SOURCE_DESC", "helpline field-test calls")

# Recording folders are named like "September 28" (no year); this year is used for the date in file_ids
FIELD_TEST_YEAR = int(os.environ.get("FIELD_TEST_YEAR", "2026"))

# Bot-language detection: if the first bot turn of a call contains one of these phrases, the call is
# labelled with that language; otherwise it gets DEFAULT_BOT_LANGUAGE. The Bhili calls mixed Bhili and
# Marathi bot greetings, so this let the two be scored separately. Phrases are the bot's scripted
# greeting, not farmer speech.
BOT_LANGUAGE_MARKERS = {"marathi": ("मी वसुधा", "नमस्कार, मी")}
DEFAULT_BOT_LANGUAGE = os.environ.get("DEFAULT_BOT_LANGUAGE", LANGUAGE.lower())
# Substring of the bot's name in ASR text; no_transcript.py uses it as a cross-check of who the bot is
BOT_NAME_ASR = os.environ.get("BOT_NAME_ASR", "सुधा")

# Label for text that came from the alignment ASR rather than a call log: transcript_source in
# utterances.csv for no-transcript clips, and the column name in work/clip_hf_asr.csv.
# Bhili used bhili_nemotron_asr, Marathi voicera_indic_asr.
ASR_LABEL = os.environ.get("ASR_LABEL", "voicera_indic_asr")

# Export issue thresholds (export_testset.py)
OVERLAP_MIN = float(os.environ.get("OVERLAP_MIN", "0.3"))   # s of bot overlap before a clip is flagged
LOW_SHARE = float(os.environ.get("LOW_SHARE", "0.3"))       # Bhili 0.1, Marathi 0.3 (long barge-in clips)


def bot_language(text):
    for lang, phrases in BOT_LANGUAGE_MARKERS.items():
        if any(p in text for p in phrases):
            return lang
    return DEFAULT_BOT_LANGUAGE
