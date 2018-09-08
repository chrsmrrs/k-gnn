#!/bin/sh

echo "1-PROTEINS"
/usr/local/miniconda3/bin/python 1-proteins.py
echo "1-PROTEINS NO TRAIN"
/usr/local/miniconda3/bin/python 1-proteins.py --no-train True
echo "1-2-3-PROTEINS"
/usr/local/miniconda3/bin/python 1-2-3-proteins.py
echo "1-2-3-PROTEINS NO TRAIN"
/usr/local/miniconda3/bin/python 1-2-3-proteins.py --no-train True
echo "1-IMDB"
/usr/local/miniconda3/bin/python 1-imdb.py
echo "1-IMDB NO TRAIN"
/usr/local/miniconda3/bin/python 1-imdb.py --no-train True
echo "1-2-3-IMDB"
/usr/local/miniconda3/bin/python 1-2-3-imdb.py
echo "1-2-3-IMDB NO TRAIN"
/usr/local/miniconda3/bin/python 1-2-3-imdb.py --no-train True
