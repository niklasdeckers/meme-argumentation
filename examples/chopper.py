from PIL import Image

from crawler import Crawler
from ocr import OCR

C = Crawler(0.002, 0.99, 10000, Image.open("examples/chopper.jpg").convert("RGB"), "chopper")

imgglob = "chopper/*.jpg"
outfile = "chopper_extractions.jsonl"
full_shape = (346, 960)
regions = [(6, 6, 340, 186),
           (2, 195, 342, 187),
           (4, 389, 338, 188),
           (4, 584, 338, 187),
           (4, 777, 339, 182)]

o = OCR(imgglob, outfile, full_shape, regions)

if __name__ == "__main__":
    for j in range(1000):
        C.crawl_step()
    o.extract_texts()
