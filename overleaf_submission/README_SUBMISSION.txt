MORPHEUS JCP submission package

Compile on Overleaf or locally with:
  latexmk -pdf main_short.tex
  latexmk -pdf supporting_information.tex
  latexmk -pdf cover_letter.tex

Included files are restricted to manuscript sources, bibliography, required generated tables, required figures, and compiled PDFs.
main_short.bbl is included for journal submission systems that request the bibliography file.
data/camphor contains the archived CSV/JSON/XYZ files used to generate the camphor validation tables and figure; these files are included for reproducibility and are not required by LaTeX at compile time.
graphical_toc.pdf and figures/graphical_toc.png provide the text-free graphical TOC at the same journal canvas used for the companion MATRIX manuscripts.

Note for submission:
  The coordinate definitions used by MORPHEUS are SMITH-generated SONIC contracts. Any internal-to-Cartesian realization is delegated to LINK; MORPHEUS does not own a separate Wilson-B inversion.
  The companion SMITH/SONIC methodological manuscript is available as arXiv:2607.16550. The standalone MORPHEUS source and reproducibility data are public at https://github.com/yogibubu/morpheus.
