# MORPHEUS JCP submission package

The submission-ready sources are kept separate from the long-form working
manuscript:

- `main_jcp.tex` / `main_jcp.pdf`: JCP-formatted Article;
- `supporting_information.tex` / `supporting_information.pdf`: separate SI;
- `cover_letter_jcp.tex` / `cover_letter_jcp.pdf`: cover letter;
- `ALT_TEXT_JCP.txt`: accessibility descriptions;
- `JCP_SUBMISSION_CHECKLIST.md`: final submission gate.

Build with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_jcp.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter_jcp.tex
```

The exact SMITH arXiv identifier is intentionally not guessed. It must be
inserted in the manuscript, bibliography, and cover letter after arXiv has
finished processing the SMITH submission.
