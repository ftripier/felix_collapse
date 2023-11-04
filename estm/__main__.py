from tap import Tap
import pathlib

from . import png
from .wvfc import WavefunctionCollapseInstance


class WvfcParser(Tap):
    source_tiles: pathlib.Path
    output: pathlib.Path


argparser = WvfcParser()
args = argparser.parse_args()

source_texture = png.load_png(args.source_tiles)
wvfc = WavefunctionCollapseInstance(source_texture)
generated_image = wvfc.run()
png.save_png(generated_image, args.output)
