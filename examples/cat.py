from PIL import Image

from crawler import Crawler
from ocr import OCR

C = Crawler(0.0012, 0.99, 10000, Image.open("examples/cat.jpg").convert("RGB"), "cat")

imgglob = "cat/*.jpg"
outfile = "cat_extractions.jsonl"
full_shape = (1009, 650)
regions = [(5, 3, 500, 165),
           (506, 3, 495, 168)]

o = OCR(imgglob, outfile, full_shape, regions)

if __name__ == "__main__":
    for j in range(1000):
        C.crawl_step()
    o.extract_texts()
