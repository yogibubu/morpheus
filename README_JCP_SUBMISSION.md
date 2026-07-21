# MORPHEUS JCP submission package

The submission-ready sources are kept separate from the long-form working
manuscript:

- `main_jcp.tex` / `main_jcp.pdf`: JCP-formatted Article;
- `supporting_information.tex` / `supporting_information.pdf`: separate SI;
- `cover_letter_jcp.tex` / `cover_letter_jcp.pdf`: cover letter;
- `JCP_TABLE_FIGURE_DESCRIPTIONS.txt`: accessibility descriptions;
- `JCP_SUBMISSION_CHECKLIST.md`: final submission gate.

Build with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_jcp.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter_jcp.tex
```

The companion SMITH/SONIC manuscript is available as arXiv:2607.16550 and is
cited in the manuscript, bibliography, and cover letter.
