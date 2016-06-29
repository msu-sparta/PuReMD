docs = README.txt

AM_V_PDFLATEX = $(AM_V_PDFLATEX_@AM_V@)
AM_V_PDFLATEX_ = $(AM_V_PDFLATEX_@AM_DEFAULT_V@)
AM_V_PDFLATEX_0 = @echo "  PDFLATEX" $@;

if BUILD_DOC
doc_DATA = manual.pdf

manual.pdf: src/manual.tex
	$(AM_V_PDFLATEX)$(PDFLATEX) $<

CLEANFILES = manual.pdf manual.log manual.out manual.aux
endif

dist_doc_DATA = ${docs}
