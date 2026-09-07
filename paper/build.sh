#!/usr/bin/env bash
# Build the AraGenre 2026 system description paper.
# Requires XeLaTeX (Arabic support) + an Arabic font, or use tectonic.
set -euo pipefail
cd "$(dirname "$0")"
MAIN=aragenre_thaka

if command -v tectonic >/dev/null 2>&1; then
  tectonic "$MAIN.tex"
elif command -v xelatex >/dev/null 2>&1; then
  xelatex "$MAIN"; bibtex "$MAIN"; xelatex "$MAIN"; xelatex "$MAIN"
else
  echo "ERROR: install tectonic (https://tectonic-typesetting.github.io) or XeLaTeX." >&2
  exit 1
fi

python3 - <<'PY'
import subprocess, sys
t = subprocess.run(["pdftotext", "aragenre_thaka.pdf", "-"], capture_output=True, text=True).stdout
pages = [p for p in t.split("\f") if p.strip()]
body_ok = any("Conclusion" in p for p in pages[:4])
print(f"pages: {len(pages)} | body within 4 pages: {body_ok} | anonymous: {'Anonymous' in pages[0]}")
sys.exit(0 if body_ok else 1)
PY
echo "OK: paper built."
