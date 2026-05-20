# Build the final report PDF from group13_v2.tex.
#
# Usage:
#   make            # build report (2 pdflatex passes for cross-refs)
#   make clean      # remove aux artifacts
#   make distclean  # also remove the PDF
#   make figures    # ensure ../outputs/figures/ is populated (regen if needed)
#
# Requires: pdflatex (TeX Live), neurips_2024.sty in this directory.

TEX        := group13_v2
PDF        := $(TEX).pdf
FIGURES    := ../outputs/figures
LATEXFLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean figures view

all: $(PDF)

$(PDF): $(TEX).tex neurips_2024.sty figures
	@echo "==> First pdflatex pass"
	pdflatex $(LATEXFLAGS) $(TEX).tex > $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@echo "==> Second pdflatex pass (cross-refs)"
	pdflatex $(LATEXFLAGS) $(TEX).tex > $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@echo "==> Built $(PDF) ($(shell pdfinfo $(PDF) 2>/dev/null | grep -i pages | awk '{print $$2}') pages)"

figures:
	@if [ ! -d $(FIGURES) ] || [ -z "$$(ls -A $(FIGURES) 2>/dev/null)" ]; then \
		echo "==> Regenerating figures from CSVs"; \
		cd .. && .venv/bin/python scripts/make_plots.py; \
		cd .. && .venv/bin/python scripts/make_robustness_plots.py; \
	else \
		echo "==> $(FIGURES) already populated ($$(ls $(FIGURES)/*.pdf 2>/dev/null | wc -l) PDFs)"; \
	fi

view: $(PDF)
	@xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || echo "Could not auto-open; PDF is at $(PDF)"

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc $(TEX).bbl $(TEX).blg \
	      $(TEX).fls $(TEX).fdb_latexmk $(TEX).synctex.gz $(TEX).build.log

distclean: clean
	rm -f $(PDF)
