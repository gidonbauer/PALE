LATEX ?= latexmk --lualatex --interaction=nonstopmode --halt-on-error --synctex=1 --output-directory=build

all: build/polar.pdf build/ALE.pdf

build/%.pdf: %.tex ${TEXSRC}
	${LATEX} $<

clean:
	rm -fr build

.PHONY: all clean
