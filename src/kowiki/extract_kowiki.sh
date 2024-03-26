#!/bin/bash
python -m wikiextractor.WikiExtractor \
       --json \
       --out ../../data/kowiki \
       ../../data/kowiki-latest-pages-meta-current.xml.bz2