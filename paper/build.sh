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

# ACL counts everything up to the unnumbered back matter against the 4-page limit.
def first_page_with(needle):
    return next((i for i, p in enumerate(pages, 1) if needle in p), None)

conclusion = first_page_with("Conclusion")
back_matter = first_page_with("Acknowledgements") or first_page_with("Limitations")
ok = conclusion is not None and conclusion <= 4 and back_matter is not None and back_matter <= 5
print(f"pages: {len(pages)} | Conclusion starts p{conclusion} | back matter starts p{back_matter}")
print("body within the 4-page limit" if ok else "BODY EXCEEDS 4 PAGES")
sys.exit(0 if ok else 1)
PY
echo "OK: paper built."
