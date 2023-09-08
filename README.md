# Usage

```
$ ./make-gif-transparent.py -h
usage: make-gif-transparent.py [-h] input_filename [output_filename]

Automatically make GIF backgrounds transparent.

positional arguments:
  input_filename   Path to the GIF
  output_filename  Path to the output GIF (default: <input-basename>-transparent.gif)

optional arguments:
  -h, --help       show this help message and exit
  --threshold THRESHOLD, -t THRESHOLD
                        The threshold for all RGB channels to determine which pixels belong to the background
```

# Installation
You just need to have ImageMagick installed (`convert` command) and the Python packages installed listed in `requirements.txt`.
```
pip install -r requirements.txt
```
