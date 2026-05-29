# Build the submission report.
#
# Usage:
#   make            # full build (pdflatex + bibtex + pdflatex x2)
#   make clean      # remove aux artifacts
#   make distclean  # also remove the PDF
#
# Requires: pdflatex, bibtex (TeX Live), neurips_2024.sty in this directory.

TEX        := main
PDF        := $(TEX).pdf
LATEXFLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(TEX).tex neurips_2024.sty refs.bib figures
	@echo "==> First pdflatex pass"
	pdflatex $(LATEXFLAGS) $(TEX).tex > $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@echo "==> bibtex pass"
	bibtex $(TEX) >> $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@echo "==> Second pdflatex pass"
	pdflatex $(LATEXFLAGS) $(TEX).tex >> $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@echo "==> Third pdflatex pass (cross-refs)"
	pdflatex $(LATEXFLAGS) $(TEX).tex >> $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@PAGES=$$(pdfinfo $(PDF) | grep -oE "Pages:.*" | awk '{print $$2}'); \
	  echo "==> Built $(PDF) ($$PAGES pages)"

figures:
	@test -d figures || (echo "ERROR: figures/ directory not found"; exit 1)

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).bbl $(TEX).blg $(TEX).out $(TEX).toc \
	      $(TEX).build.log $(TEX).fls $(TEX).fdb_latexmk $(TEX).synctex.gz

distclean: clean
	rm -f $(PDF)
