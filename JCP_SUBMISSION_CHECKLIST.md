# MORPHEUS — JCP submission checklist

## Ready

- [x] JCP/AIP REVTeX 4.2 two-column manuscript (`aip,jcp,reprint`).
- [x] One-paragraph abstract below the 250-word limit (216 words) and free of
  technical acronyms apart from the method name MORPHEUS.
- [x] Article type identified as **Article** in the cover letter.
- [x] Separate compiled main-manuscript and Supporting Information PDFs.
- [x] Conflict-of-interest, CRediT author-contribution, and Data Availability statements.
- [x] Proper REVTeX captions, labels, numbering, and in-text references for all main figures and tables.
- [x] Separate alt-text file prepared for the main manuscript and Supporting Information figure (`JCP_TABLE_FIGURE_DESCRIPTIONS.txt`).
- [x] Representative input and output shown explicitly in the Supporting Information.
- [x] Revision-pinned standalone source, runnable examples, benchmark inputs,
  and machine-readable validation data published at
  <https://github.com/yogibubu/morpheus>.
- [x] Main manuscript, Supporting Information, and cover letter compile without unresolved references, citations, or overfull boxes.
- [x] All pages visually inspected after PDF rendering.

## Final gate after SMITH processing

- [x] Insert the definitive SMITH/SONIC citation (arXiv:2607.16550) in `references.bib` and `main_jcp.tex`.
- [x] Replace the temporary arXiv-processing sentence in `cover_letter_jcp.tex` with the definitive citation.
- [x] Recompile all PDFs and repeat the warning and visual checks.
- [ ] Confirm author metadata, suggested reviewers, and any excluded reviewers in the JCP submission portal.
- [ ] Upload `main_jcp.pdf` as the manuscript and `supporting_information.pdf` as a separate supplementary file.
- [ ] Perform the final author proof of the portal-generated submission PDF before approval.

Current AIP instructions: <https://publishing.aip.org/resources/researchers/author-instructions/>
