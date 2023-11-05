from tap import Tap
import pathlib

from . import png
from .wvfc import WavefunctionCollapse


class WvfcParser(Tap):
    source_tiles: pathlib.Path
    output: pathlib.Path
    height: int = 100
    width: int = 100


argparser = WvfcParser()
args = argparser.parse_args()

source_texture = png.load_png(args.source_tiles)
wvfc = WavefunctionCollapse(source_texture)
print(f"Running with args {args}")
generated_image = wvfc.run((args.height, args.width))
png.save_png(generated_image, args.output)
