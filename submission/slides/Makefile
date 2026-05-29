# Build the presentation slides.
# Usage: make
TEX        := slides
PDF        := $(TEX).pdf
LATEXFLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(TEX).tex
	pdflatex $(LATEXFLAGS) $(TEX).tex > $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	pdflatex $(LATEXFLAGS) $(TEX).tex >> $(TEX).build.log 2>&1 || (tail -30 $(TEX).build.log; exit 1)
	@PAGES=$$(pdfinfo $(PDF) | grep -oE "Pages:.*" | awk '{print $$2}'); \
	  echo "==> Built $(PDF) ($$PAGES slides)"

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).nav $(TEX).out $(TEX).snm $(TEX).toc $(TEX).vrb $(TEX).build.log

distclean: clean
	rm -f $(PDF)
