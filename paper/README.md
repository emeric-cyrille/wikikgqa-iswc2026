# Paper — WikiKGQA @ ISWC 2026

Two independent LaTeX sources of the same paper (identical content, identical
data, identical figures). Choose the one matching the venue's language.

| Folder | Language | PDF                          |
|--------|----------|------------------------------|
| `fr/`  | French   | [`fr/main.pdf`](fr/main.pdf) |
| `en/`  | English  | [`en/main.pdf`](en/main.pdf) |

## Compilation

Both sources build with **pdfLaTeX** using the CEURART class shipped alongside.
Required TeX Live packages: `libertinus`, `libertinus-type1`, `libertinust1math`,
`fontaxes`. From either directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Fonts

The CEURART class already loads Libertinus Serif, Sans and Mono. We add
`libertinust1math` in the preamble so that mathematics render in Libertinus Math
rather than in Computer Modern.
