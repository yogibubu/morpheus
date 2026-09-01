MORPHEUS JCP submission package

Compile on Overleaf or locally with:
  latexmk -pdf revised_marked.tex
  latexmk -pdf revised_clean.tex
  latexmk -pdf response_letter.tex
  latexmk -pdf supporting_information.tex
  latexmk -pdf cover_letter.tex

Included files are restricted to manuscript sources, bibliography, required generated tables, required figures, and compiled PDFs.
revised_marked.bbl is included for journal submission systems that request the bibliography file.
data/camphor contains the archived CSV/JSON/XYZ files used to generate the camphor validation tables and figure; these files are included for reproducibility and are not required by LaTeX at compile time.
graphical_toc.pdf and figures/graphical_toc.png provide the text-free graphical TOC.

Note for submission:
  The coordinate definitions used by MORPHEUS are SMITH-generated SONIC contracts. Internal-to-Cartesian realization uses a rank-controlled minimum-norm or nonlinear procedure; MORPHEUS does not own a separate Wilson-B inversion.
  The companion SMITH/SONIC methodological manuscript is available as arXiv:2607.16550. The standalone MORPHEUS source and reproducibility data are public at https://github.com/yogibubu/morpheus.

Referee-response page map (revised_marked.pdf): computational-overhead paragraph, p. 11; Figure 1, p. 3; Figure 5, p. 13. The same page map applies to the clean manuscript.
The point-by-point response is provided as response_letter.tex and response_letter.pdf.
