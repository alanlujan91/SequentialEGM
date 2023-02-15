#!/bin/bash
make4ht --utf8 --config EGMN.cfg --format html5 EGMN "svg" "-cunihtf -utf8"
cat page-style.css | cat - EGMN-generated-by-make4ht.css > EGMN-page-style.css && mv EGMN-page-style.css EGMN.css
cp EGMN.html index.html
