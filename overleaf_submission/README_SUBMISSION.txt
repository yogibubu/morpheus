MORPHEUS JCP submission package

Compile on Overleaf or locally with:
  latexmk -pdf main_short.tex
  latexmk -pdf supporting_information.tex
  latexmk -pdf cover_letter.tex

Included files are restricted to manuscript sources, bibliography, required generated tables, required figures, and compiled PDFs.
main_short.bbl is included for journal submission systems that request the bibliography file.
data/camphor contains the archived CSV/JSON/XYZ files used to generate the camphor validation tables and figure; these files are included for reproducibility and are not required by LaTeX at compile time.

Note for submission:
  The coordinate definitions and Wilson B matrices used by MORPHEUS are SMITH-generated SONIC coordinate contracts.
  The companion SMITH/SONIC methodological manuscript is in preparation for publication and can be uploaded as confidential supplementary material for referees if the journal requests it.
