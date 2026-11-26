"""Local server for the manual-fix page. Standard library only.

  python3 scripts/manual_fix/server.py [port]      (default 8765, binds 127.0.0.1)

Then open http://localhost:8765 (in VS Code Remote, the port is forwarded automatically;
otherwise use the Ports panel). Serves work/manual/ and saves every decision to
work/manual/fixes.csv (one row per utt_id; saving again overwrites that row).
"""
import csv, json, os, sys, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(ROOT, "work", "manual")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
FIXES = os.path.join(WORK, "fixes.csv")
FIELDS = ["utt_id", "decision", "start_s", "end_s", "win_start", "note", "reviewer", "saved_at"]
DECISIONS = {"farmer", "not_in_recording", "bot_only", "unsure"}
LOCK = threading.Lock()


def load():
    if not os.path.exists(FIXES):
        return {}
    return {r["utt_id"]: r for r in csv.DictReader(open(FIXES, encoding="utf-8"))}


def save(rows):
    tmp = FIXES + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: r["utt_id"]))
    os.replace(tmp, FIXES)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WORK, **k)

    def log_message(self, fmt, *args):      # keep the terminal quiet except for saves
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = open(PAGE, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/fixes":
            with LOCK:
                self._json(load())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/fix":
            return self._json({"error": "not found"}, 404)
        try:
            r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            assert r.get("decision") in DECISIONS, "bad decision"
            assert r.get("utt_id"), "missing utt_id"
            if r["decision"] == "farmer":
                assert float(r["end_s"]) > float(r["start_s"]) >= 0, "bad span"
            row = {k: str(r.get(k, "") if r.get(k) is not None else "") for k in FIELDS}
        except (ValueError, AssertionError, TypeError) as e:
            return self._json({"error": str(e)}, 400)
        with LOCK:
            rows = load()
            rows[row["utt_id"]] = row
            save(rows)
        print(f"saved {row['utt_id']}: {row['decision']} {row['start_s']}-{row['end_s']}", flush=True)
        self._json({"ok": True, "done": len(rows)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"manual fix page: http://localhost:{port}   (decisions -> {FIXES})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
