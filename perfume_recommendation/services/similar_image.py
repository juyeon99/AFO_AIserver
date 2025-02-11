from sqlalchemy.orm import Session
import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, resnet50
from PIL import Image
import requests
from functools import lru_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .db_service import Product, ProductImage
from perfume_recommendation.embedding_utils import save_embedding, load_embedding  # ✅ 캐시 불러오기 추가


# ✅ 이미지 모델 설정
IMAGE_MODEL_TYPE = "vit"
IMAGE_MODEL_CONFIG = {
    "vit": vit_b_16,
    "resnet": resnet50,
}

image_model = IMAGE_MODEL_CONFIG[IMAGE_MODEL_TYPE](pretrained=True)
image_model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def get_similar_image_embedding(image_url: str):
    """이미지 임베딩을 캐시에서 불러오거나, 새로 계산"""
    
    # ✅ 먼저 캐시에서 불러오기 시도
    cached_embedding = load_embedding(image_url)
    if cached_embedding is not None:
        print(f"✅ 캐시에서 임베딩 불러옴: {image_url}")
        return cached_embedding  # 캐시가 있으면 바로 반환

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = image_model(image)
            if isinstance(embedding, tuple):
                embedding = embedding[0]

        embedding = embedding.numpy().flatten()

        # ✅ 새로 계산된 임베딩을 저장 (JSON 캐싱)
        save_embedding(image_url, embedding)
        return embedding

    except Exception:
        return None

def find_similar_images(product_id: int, db: Session, top_n: int = 5):
    """이미지 기반 유사 향수 검색"""
    try:
        target_image = (
            db.query(ProductImage).filter(ProductImage.product_id == product_id).first()
        )
        if not target_image:
            return []

        target_embedding = get_similar_image_embedding(target_image.url)
        if target_embedding is None:
            return []

        # 모든 향수 이미지 가져오기
        all_images = (
            db.query(ProductImage)
            .join(Product, ProductImage.product_id == Product.id)
            .filter(Product.category_id == 1)
            .filter(Product.id != product_id)
            .distinct(ProductImage.product_id)
            .all()
        )

        # 제품 ID별로 최고 유사도 저장
        product_similarities = {}
        for img in all_images:
            try:
                img_embedding = get_similar_image_embedding(img.url)
                if img_embedding is None:
                    continue

                similarity = cosine_similarity([target_embedding], [img_embedding])[0][
                    0
                ]

                # 같은 제품이면 더 높은 유사도로 업데이트
                if (
                    img.product_id not in product_similarities
                    or similarity > product_similarities[img.product_id]
                ):
                    product_similarities[img.product_id] = similarity

            except Exception:
                continue

        # 유사도 기준으로 상위 N개 선택
        sorted_similarities = [
            {"product_id": pid, "similarity": sim}
            for pid, sim in sorted(
                product_similarities.items(), key=lambda x: x[1], reverse=True
            )
        ][:top_n]

        return sorted_similarities

    except Exception:
        return []
