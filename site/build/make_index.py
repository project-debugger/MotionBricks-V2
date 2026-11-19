#!/usr/bin/env python3
"""Write site/index.html (GitHub Pages, full document) and build/artifact/index.html (claude.ai body-only)."""
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
src = (SITE / "index.src.html").read_text()
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">\n'
        '<meta name="theme-color" content="#10161e">\n'
        '<!-- enables SharedArrayBuffer (multi-threaded WASM) on hosts without COOP/COEP headers, e.g. GitHub Pages -->\n'
        '<script src="coi-serviceworker.js"></script>\n')
title_end = src.index("</style>") + len("</style>")
full = head + src[:title_end] + "\n</head>\n<body>\n" + src[title_end:] + "\n</body>\n</html>\n"
(SITE / "index.html").write_text(full)
art = SITE / "build" / "artifact"
art.mkdir(parents=True, exist_ok=True)
# claude.ai artifacts accept only standard web media types: the binary parts are published as .wasm (same bytes)
(art / "index.html").write_text('<meta name="soma-bin-ext" content=".wasm">\n' + src)
print("wrote site/index.html and site/build/artifact/index.html")
