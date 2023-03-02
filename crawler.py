import clip
import faiss
import io
import numpy as np
import os
import queue
import torch
import urllib.request
from PIL import Image as pimage
from clip_retrieval.clip_client import ClipClient, Modality
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return pimage.open(img_stream).convert("RGB")


class Crawler:
    def __init__(self, similarity_threshold, duplicate_threshold, max_queue_size, initial_image, out_dir,
                 knn_service_url="https://knn.laion.ai/knn-service"):

        self.model, self.preprocess = clip.load("ViT-L/14", device="cuda", jit=True)

        self.client = ClipClient(
            url=knn_service_url,
            indice_name="laion5B-L-14",
            aesthetic_score=9,
            aesthetic_weight=0.5,
            modality=Modality.IMAGE,
            num_images=40,
            deduplicate=True,
            use_safety_model=False,
            use_violence_detector=False
        )
        self.similarity_threshold = similarity_threshold
        self.q = queue.Queue(maxsize=max_queue_size)  # todo multiprocessing queue
        initial_emb = self.get_image_emb(initial_image)
        self.initial_image_np = np.array(initial_image).astype("float64")
        self.q.put(initial_emb)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.i = 0  # todo multiprocessing safe?
        self.index = faiss.IndexFlatIP(initial_emb.shape[-1])
        self.register_embedding(initial_emb)
        self.duplicate_threshold = duplicate_threshold

    def check_similarity(self, img):
        img_np = np.array(img)
        image_resized = resize(img_np, self.initial_image_np.shape[:2])
        similarity = ssim(self.initial_image_np, np.array(image_resized), channel_axis=-1)
        self.deb_sim = similarity  # todo remove
        return similarity >= self.similarity_threshold

    def get_image_emb(self, image):
        with torch.no_grad():
            image_emb = self.model.encode_image(self.preprocess(image).unsqueeze(0).to("cpu"))
            image_emb /= image_emb.norm(dim=-1, keepdim=True)
            image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
            return image_emb

    def save_img(self, img):
        img.save(os.path.join(self.out_dir, f"img_{self.i:05}.jpg"))
        self.i += 1

    def detect_duplicate(self, emb):
        inner_product, neighbor = self.index.search(emb.reshape([1, -1]), 1)
        self.deb_ip, self.deb_n = inner_product.item(), neighbor.item()  # todo remove
        return inner_product.item() >= self.duplicate_threshold

    def register_embedding(self, emb):
        self.index.add(emb.reshape([1, -1]))

    def crawl_step(self):
        try:
            emb = self.q.get(block=False)
        except queue.Empty:
            return
        children = self.client.query(embedding_input=emb.tolist())
        for child in children:
            try:
                img = download_image(child["url"])
            except:
                continue
            if not self.check_similarity(img):
                continue
            child_emb = self.get_image_emb(img)
            if self.detect_duplicate(child_emb):
                continue
            self.register_embedding(child_emb)
            print(f"i {self.i:05} - SIM {self.deb_sim:.5f} - IP {self.deb_ip:.5f} - N {self.deb_n:05}")  # todo remove
            self.save_img(img)
            try:
                self.q.put(child_emb, block=False)
            except queue.Full:
                continue


if __name__ == "__main__":
    C = Crawler(0.0007, 0.99, 10000, pimage.open("chopper.jpg").convert("RGB"), "chopper")
    for j in range(1000):
        C.crawl_step()
