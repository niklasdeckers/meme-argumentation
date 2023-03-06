from PIL import Image

from crawler import Crawler
from ocr import OCR

C = Crawler(0.00015, 0.99, 10000, Image.open("examples/rock.jpg").convert("RGB"), "rock")

imgglob = "rock/*.jpg"
outfile = "rock_extractions.jsonl"
full_shape = (568, 700)
regions = [(2, 3, 563, 228),
           (4, 235, 561, 229),
           (0, 468, 566, 229)]

o = OCR(imgglob, outfile, full_shape, regions)

if __name__ == "__main__":
    for j in range(1000):
        C.crawl_step()
    o.extract_texts()
