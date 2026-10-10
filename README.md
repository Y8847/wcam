# WCAM — print-cam robust watermarking

[Русская версия](README.ru.md)

An implementation of Pramila, Keskinarkaus and Seppänen, "Increasing the
capturing angle in print-cam robust watermarking" (Journal of Systems and
Software 135, 2018), together with the embedding details from the same
authors' earlier paper in Signal, Image and Video Processing 6 (2012).

A 32-bit message is encoded in the rotation angles of a periodic pattern added
to sixteen blocks of an image. The image is printed, photographed with a phone,
possibly at an angle, and the message is read back without the original image.

This repository contains the desktop (Python) part only. The iPhone app that
recorded the focal stacks is not included; its output format is documented
below, and one example capture is provided, so the focal-stack pipeline can
still be run.

Script output and code comments are mostly in Russian.

## Installation

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

Python 3.10 or newer is required; tested on 3.14.7.

## Embed a message

    python embed_cli.py sample_images/peppers.png --payload deadbeef

This writes `out/peppers_wm.png`, the file to print or show on a screen. Next
to it go `out/peppers_wm_core.png` (the image without the frame) and
`out/peppers_wm.json` with the payload and the fifteen embedded angles, which
can later be used to check a reading. Without `--payload` the payload is
random. `--delta1` and `--delta2` set the embedding strength (100 and 10 by
default, as in the paper), `--out` sets the output name.

The pattern is built by repeating a small pseudorandom tile. A different tile
can be given with `--tile`:

    python embed_cli.py sample_images/peppers.png --tile tile_demo.npy
    python embed_cli.py sample_images/peppers.png --tile tile22_fixed.npy

Either a `.npy` array or an image works; in an image, light reads as +1 and
dark as −1. Any size is accepted, not only 28×7. Nothing else changes: the
rotation, the strength and the visibility map stay the same.

Two tiles are included: `tile22_fixed.npy`, recovered from the authors'
published figures, and `tile_demo.npy`, an arbitrary 12×5 example.

The reader does not need the tile: the lattice angle is measured by
autocorrelation, which does not depend on the tile's contents. The tile can
therefore be changed without touching `process_photo.py`.

`--mode` selects how closely to follow the authors' published images:
`plain` (the default) uses our tile with bilinear rotation, as in Eq. 14 of
the 2012 paper; `papercalib` and `papermode` use their tile with
nearest-neighbour rotation, with and without amplitude calibration.

## Read a message from a single photo

    python process_photo.py out/peppers_wm.png

Finds the frame, corrects the perspective and measures the angle of each
block. Prints the share of wrong blocks before error correction and compares
the decoded payload with the one stored in the metadata.

The metadata is optional: the message is read from the photo itself, and the
reference is only used to check the answer. The file is looked up next to the
photo by name, then in `out/`. Without it the reading still happens:

    python process_photo.py photo.jpg
    decoded payload: deadbeef

The decoded payload is checked by re-encoding it. Reed–Solomon does not
report failure: given more errors than it can correct, it still returns a
plausible payload. So the answer is encoded again and compared with the
measured angles. A real correction disagrees with them in at most three
places, a false one in many more. Without a reference this is the only way
to tell a read message from an invented one.

Run without arguments to print usage.

## Read a message from a focal stack

    python _run_real_stack.py <capture directory>

The full pipeline of Sections 3.1–3.8: sharpness maps, frame selection,
rescaling, AKAZE registration, blending, rectification, block reading. The
input is a capture directory containing `stack.bin` with the frames and
`notes.txt` with the capture parameters. Such directories were recorded by
our iPhone app, which is not part of this repository; any tool that writes
the `stack.bin` format described below can produce them.

An example capture is included at the top level of this repository:

    python _run_real_stack.py .

Run without arguments to print usage.

## Split a focal stack into frames

    python dump_stack.py <capture directory or stack.bin>

Writes each frame of the stack as a separate PNG into `out/<capture name>/`
and prints a table of frame number, lens position and focus distance. Useful
for seeing what the camera actually captured, for example why the sheet is
sharp in one frame and not in another. A second argument sets a different
output directory. Run without arguments to print usage.

## HEIC

iPhones shoot HEIC, which OpenCV cannot read. The scripts handle it
themselves: they use `pillow-heif` if it is installed, otherwise `heif-dec`
from libheif. A photo can be passed as is:

    python embed_cli.py IMG_2399.HEIC --payload cafe1234
    python process_photo.py out/IMG_2399_wm.png

If neither is available, the script says so and suggests converting the file
to PNG or JPEG.

## Output

Everything the scripts produce goes into `out/` next to the scripts. A name
with a path, for example `--out /tmp/sheet`, is used as given.

## The stack.bin format

The header is four little-endian 32-bit integers: frame count, width, height
and a brightness scale. The frames follow one after another, `width × height`
unsigned 16-bit integers each, row by row. A pixel value is the luminance
multiplied by the scale (usually 64); there is one channel.

## Files

| file | purpose |
| --- | --- |
| `pattern.py` | the 28×7 pseudorandom tile and its rotation (Section 2) |
| `jnd.py` | the just-noticeable-difference map, after Chou and Li with the authors' modification |
| `preprocess.py` | Wiener filtering before embedding (Eq. 1) |
| `codec.py` | RS(15,8) over GF(16), Gray code, angles (Eq. 11) |
| `embedder.py` | the embedding itself (Eq. 15) |
| `embed_cli.py` | command line for embedding |
| `sharpness.py` | focal-stack sharpness maps (3.2) |
| `substack.py` | frame selection (3.3) |
| `reconfigure.py` | rescaling (3.4) |
| `register.py` | AKAZE registration (3.5) |
| `blend.py` | blending (3.6) |
| `rectify.py` | frame detection and rectification (3.7) |
| `extractor.py` | block angle measurement (3.8) |
| `process_photo.py` | reading a single photo |
| `_run_real_stack.py` | reading a focal stack |
| `capture.py`, `degrade.py` | synthetic print-and-capture model |
| `imread.py` | image reading, including HEIC |
| `dump_stack.py` | splitting stack.bin into frames |

Section and equation numbers refer to the 2018 paper unless stated otherwise.

## Sample data

`sample_images/` holds the Baboon and Peppers test images. `stack.bin` and
`notes.txt` are a real focal-stack capture of a printed watermarked sheet.

## License

MIT, see [LICENSE](LICENSE). The license covers this code; the method itself
is described in the papers cited above.
