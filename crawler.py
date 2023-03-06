import io
import os
import queue
import urllib.request

import clip
import faiss
import numpy as np
import torch
from PIL import Image
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
    return Image.open(img_stream).convert("RGB")


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
        self.q = queue.Queue(maxsize=max_queue_size)
        initial_emb = self.get_image_emb(initial_image)
        self.initial_image_np = np.array(initial_image).astype("float64")
        self.q.put(initial_emb)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.i = 0
        self.index = faiss.IndexFlatIP(initial_emb.shape[-1])
        self.register_embedding(initial_emb)
        self.duplicate_threshold = duplicate_threshold

    def get_similarity(self, img):
        img_np = np.array(img)
        image_resized = resize(img_np, self.initial_image_np.shape[:2])
        similarity = ssim(self.initial_image_np, np.array(image_resized), channel_axis=-1)
        return similarity

    def get_image_emb(self, image):
        with torch.no_grad():
            image_emb = self.model.encode_image(self.preprocess(image).unsqueeze(0).to("cpu"))
            image_emb /= image_emb.norm(dim=-1, keepdim=True)
            image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
            return image_emb

    def save_img(self, img, filename):
        img.save(os.path.join(self.out_dir, filename))
        self.i += 1

    def detect_duplicate(self, emb):
        inner_product, neighbor = self.index.search(emb.reshape([1, -1]), 1)
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
            similarity = self.get_similarity(img)
            if not similarity >= self.similarity_threshold:
                continue
            child_emb = self.get_image_emb(img)
            if self.detect_duplicate(child_emb):
                continue
            self.register_embedding(child_emb)
            self.save_img(img, f"img_{self.i:05}_{similarity:.5f}.jpg")
            try:
                self.q.put(child_emb, block=False)
            except queue.Full:
                continue
