#!/usr/bin/env python3
"""Write site/index.html (the full page served by GitHub Pages) from site/index.src.html."""
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
print("wrote site/index.html")
