from PIL import Image

from crawler import Crawler
from ocr import OCR

C = Crawler(0.00023, 0.99, 10000, Image.open("examples/boyfriend.jpg").convert("RGB"), "boyfriend")

imgglob = "boyfriend/*.jpg"
outfile = "boyfriend_extractions.jsonl"
full_shape = (750, 500)
regions = [(20, 82, 339, 405),
           (343, 53, 224, 429),
           (551, 45, 194, 449)]

o = OCR(imgglob, outfile, full_shape, regions)

if __name__ == "__main__":
    for j in range(1000):
        C.crawl_step()
    o.extract_texts()
