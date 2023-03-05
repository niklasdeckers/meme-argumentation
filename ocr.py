import glob
import json
import typing
from dataclasses import dataclass

import pytesseract
from PIL import Image


def extract_text(img):
    return pytesseract.image_to_string(img, timeout=2).strip()


@dataclass
class OCR:
    imgglob: str
    outfile: str
    full_shape: typing.Tuple[int]
    regions: typing.List[typing.Tuple[int]]

    def extract_texts(self):
        with open(self.outfile, "w") as f:
            for file in sorted(glob.glob(self.imgglob)):
                img = Image.open(file)
                resultlist = [file]
                for region in self.regions:
                    x1 = region[0] / self.full_shape[0] * img.size[0]
                    x2 = x1 + region[2] / self.full_shape[0] * img.size[0]
                    y1 = region[1] / self.full_shape[1] * img.size[1]
                    y2 = y1 + region[3] / self.full_shape[1] * img.size[1]
                    img_part = img.crop((int(x1), int(y1), int(x2), int(y2)))
                    resultlist.append(extract_text(img_part))
                json.dump(resultlist, f)
                f.write("\n")


if __name__ == "__main__":
    imgglob = "chopper/*.jpg"
    outfile = "chopper_extractions.jsonl"
    full_shape = (346, 960)  # image_width, image_height
    # bbox_x, bbox_y, bbox_width, bbox_height
    # use a tool like https://www.makesense.ai/ to extract the following rect annotations
    regions = [(6, 6, 340, 186),
               (2, 195, 342, 187),
               (4, 389, 338, 188),
               (4, 584, 338, 187),
               (4, 777, 339, 182)]
    o = OCR(imgglob, outfile, full_shape, regions)
    o.extract_texts()
