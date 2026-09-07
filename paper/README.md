# Thakaa at AraGenre 2026: system description paper

Source for the ArabicNLP 2026 system description paper of the 1st-place AraGenre system
(0.7352 Hierarchical Macro F1).

## Build

```bash
make                # or: ./build.sh   -- uses tectonic if available, else XeLaTeX
make clean          # remove generated LaTeX files
```

Produces `aragenre_thaka.pdf` and checks that the main body fits the 4-page limit.

## Files

| File | Purpose |
|---|---|
| `aragenre_thaka.tex` | paper source (ACL format, XeLaTeX for Arabic examples) |
| `aragenre.bib` | bibliography (34 references) |
| `acl.sty`, `acl_natbib.bst` | official ACL style, unmodified |
| `aragenre_thaka.pdf` | compiled paper |

## Notes

- Compile with **XeLaTeX or tectonic**, not pdfLaTeX: the paper embeds Arabic examples via
  `polyglossia`/`fontspec`. The Arabic font is set with `\newfontfamily\arabicfont`
  (currently Noto Sans Arabic); change that line if the font is unavailable.
- Do not modify `acl.sty`; non-conforming formatting is rejected without review.
- The paper is **not anonymous** (system description papers are not anonymised).
- Page limit: 4 pages of body; references, appendix, acknowledgements, limitations and
  ethics statements do not count.
