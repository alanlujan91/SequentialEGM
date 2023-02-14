#!/bin/bash
make4ht --utf8 --config HAFiscal.cfg --format html5 HAFiscal "svg" "-cunihtf -utf8"
cat page-style.css | cat - HAFiscal-generated-by-make4ht.css > HAFiscal-page-style.css && mv HAFiscal-page-style.css HAFiscal.css
cp HAFiscal.html index.html
