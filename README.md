    # keyword-spotting
This repository contains my work on keyword spotting using game theory.


## Installation
On OS X, simply use `brew install opencv3 --with-contrib --with-python3` to install OpenCV, then `echo /usr/local/opt/opencv3/lib/python3.4/site-packages >> /usr/local/lib/python3.4/site-packages/opencv3.pth` to set up correctly the Python path.


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


## License
The MIT License (MIT)

Copyright (c) 2015 Francesco Cagnin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
