"""Optional: map each downloaded raw file to its Google Drive file id, so export_testset.py can put
Drive links (recording_drive_url, transcript_drive_url) in metadata.csv.

Input: one or more JSON listings of the Drive folders, as a list of {"url": ".../uc?id=<id>", "path": "<name>"}
(the format gdown prints for a folder with --json; save stdout to a file per folder).
Files are matched by base name to the files under raw/, so names must be unique across folders.

Writes work/drive_ids.json = {"<path relative to raw/>": "<drive id>"}.

Usage: python3 scripts/make_drive_ids.py raw work/drive_listing_recordings.json work/drive_listing_transcripts.json
"""
import json, os, re, sys, collections


def main(raw, *listings):
    by_name, dup = {}, set()
    for f in listings:
        for e in json.load(open(f, encoding="utf-8")):
            m = re.search(r"[?&]id=([\w-]+)", e["url"]) or re.search(r"/d/([\w-]+)", e["url"])
            name = os.path.basename(e["path"])
            if m:
                if name in by_name and by_name[name] != m.group(1):
                    dup.add(name)
                by_name[name] = m.group(1)
    ids, missing = {}, collections.Counter()
    for d, _, files in os.walk(raw):
        for n in files:
            rel = os.path.relpath(os.path.join(d, n), raw)
            if n in dup:
                missing["ambiguous name"] += 1
            elif n in by_name:
                ids[rel] = by_name[n]
            else:
                missing["not in listing"] += 1
    os.makedirs("work", exist_ok=True)
    json.dump(ids, open("work/drive_ids.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{len(ids)} files mapped -> work/drive_ids.json; unmapped: {dict(missing) or 0}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], *sys.argv[2:])
