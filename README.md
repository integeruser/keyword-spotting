# keyword-spotting
This repository contains a few unfinished python scripts for keywords detection on handwritten documents.  
The data set used for early experiments can be found [here](http://ciir.cs.umass.edu/downloads/old/data_sets.html).  
Check the Downloads page for more unfinished scripts.


## Installation
On OS X, simply use `brew install opencv3 --c++11 --with-contrib --with-python3 --with-qt` to install OpenCV, followed by `echo /usr/local/opt/opencv3/lib/python3.4/site-packages >> /usr/local/lib/python3.4/site-packages/opencv3.pth` to set up correctly the Python path.


## Usage
```
keyword-spotting ➤ cat args.json
{
    "pages_directory_path": "../data sets/typesetted",
    "contrast_threshold": "0.04",
    "n_octave_layers": "1",

    "codebook_size": "256",
    "max_iter": "10",
    "epsilon": "1.0",

    "query_file_path": "../queries/typesetted/et.png",
    "n_features": "20",
    "rho": "0.9"
}
keyword-spotting ➤ mkdir /tmp/output
keyword-spotting ➤ python3 -Bu main.py -o /tmp/output args.json
```