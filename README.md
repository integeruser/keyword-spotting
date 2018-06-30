# keyword-spotting
This repository contains a few (unfinished) Python scripts for keyword detection on handwritten documents; the data set used for experimenting consisted of [20 pages of George Washington's manuscripts](http://ciir.cs.umass.edu/downloads/old/data_sets.html).

## Requisites
- OpenCV 3 (with Python bindings)

## Usage
0. Download and unzip the aforementioned data set:
```
$ tar xf gw_20p_wannot.tgz
$ ls gw_20p_wannot/
2700270.tif         2740274.tif         2780278.tif         3020302.tif         3060306.tif         Changelog.txt
2700270_boxes.txt   2740274_boxes.txt   2780278_boxes.txt   3020302_boxes.txt   3060306_boxes.txt   README.txt
2710271.tif         2750275.tif         2790279.tif         3030303.tif         3070307.tif         annotations.txt
2710271_boxes.txt   2750275_boxes.txt   2790279_boxes.txt   3030303_boxes.txt   3070307_boxes.txt   copyrightnotice.txt
2720272.tif         2760276.tif         3000300.tif         3040304.tif         3080308.tif         file_order.txt
2720272_boxes.txt   2760276_boxes.txt   3000300_boxes.txt   3040304_boxes.txt   3080308_boxes.txt   seg_boxes.m
2730273.tif         2770277.tif         3010301.tif         3050305.tif         3090309.tif
2730273_boxes.txt   2770277_boxes.txt   3010301_boxes.txt   3050305_boxes.txt   3090309_boxes.txt
```
1. Construct the corpus:
```
$ python3 src/1-construct-corpus.py ~/gw_20p_wannot/ 0.03 1
Processing: 2740274.tif
Processing: 3080308.tif
. . .
Processing: 2700270.tif
Saving: corpus-f9af424.pickle
```
2. Construct the codebook:
```
$ python3 src/2-construct-codebook.py ./corpus-f9af424.pickle 256 10 1.0
Loading: ./corpus-f9af424.pickle
Clustering the descriptors space...
Saving: codebook-e1e2faa.pickle
```
3. Find matches against a query:
```
$ python3 src/3-find-matches.py ./codebook-e1e2faa.pickle ./query-to.png 10 0.5
Query keypoints count: 10
Query keypoints count after pruning: 8
A match must contain at least 4 keypoints (since rho=0.5)
Loading: ./codebook-e1e2faa.pickle
Codebook keypoints count: 313549
Finding nearest codewords...
Finding matches...
0 matches in page 2740274.tif
0 matches in page 3080308.tif
1 matches in page 2780278.tif
. . .
6 matches in page 3000300.tif
1 matches in page 2700270.tif
Saving: matches-135df12.json
```
4. Draw matches on the manuscript pages:
```
$ python3 src/4-draw-matches.py ~/gw_20p_wannot/ ./matches-135df12.json
Loading: ./matches-135df12.json
Extracting top-20 matches in each page...
Saving: /var/folders/5q/0gm82km97cs3p1r4t0br0d940000gn/T/tmpb6ge_hei/2780278.tif.png
Saving: /var/folders/5q/0gm82km97cs3p1r4t0br0d940000gn/T/tmpb6ge_hei/3040304.tif.png
. . .
Saving: /var/folders/5q/0gm82km97cs3p1r4t0br0d940000gn/T/tmpb6ge_hei/2700270.tif.png
```
