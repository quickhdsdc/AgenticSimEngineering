"""Loopback-only static server exposing verified bundle files."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

from .bundle import verify


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory, allowed, **kwargs):
        self.allowed = allowed
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        name = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
        if name not in self.allowed:
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        name = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
        if name not in self.allowed:
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        print(format % args, file=sys.stderr)


def server(directory, port=8765):
    directory = Path(directory).resolve()
    manifest = verify(directory)
    handler = partial(ViewerHandler, directory=str(directory), allowed=set(manifest["files"]) | {"bundle.json"})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(directory, port):
    with server(directory, port) as httpd:
        print(json.dumps({"status": "serving", "url": f"http://127.0.0.1:{httpd.server_port}/",
                          "bundle": str(Path(directory).resolve())}), flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
