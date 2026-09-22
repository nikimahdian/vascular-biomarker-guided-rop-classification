#!/bin/sh
set -eu

# latexmk runs XeLaTeX as many times as needed to resolve the table of
# contents, figure/table references, citations, and page numbers.
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
