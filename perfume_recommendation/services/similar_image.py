from sqlalchemy.orm import Session
import torch
from torchvision.models import vit_b_16, swin_v2_b, Swin_V2_B_Weights
from transformers import ConvNextModel, ConvNextImageProcessor
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
from .db_service import Product, ProductImage
from perfume_recommendation.embedding_utils import save_embedding, load_embedding
import logging

logger = logging.getLogger(__name__)

# ✅ 이미지 모델 설정
IMAGE_MODEL_TYPE = "convnext"  # 'convnext', 'swin', 'vit' 중 하나

# ✅ 모델 및 전처리기 초기화
if IMAGE_MODEL_TYPE == "convnext":
    model_path = "facebook/convnext-base-224"
    image_model = ConvNextModel.from_pretrained(model_path)
    image_processor = ConvNextImageProcessor.from_pretrained(model_path)
elif IMAGE_MODEL_TYPE == "swin":
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    image_model = swin_v2_b(weights=weights)
    image_processor = weights.transforms()
else:  # vit
    image_model = vit_b_16(pretrained=True)
    image_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224")  # 통합 사용

image_model.eval()


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

        # ✅ 전처리 (모든 모델에서 동일하게 image_processor 활용)
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            if IMAGE_MODEL_TYPE == "convnext":
                outputs = image_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            elif IMAGE_MODEL_TYPE == "swin":
                # ✅ Swin Transformer의 feature extraction 방식 수정
                features = image_model.forward_features(inputs["pixel_values"])
                embedding = features.mean(dim=[2, 3]).squeeze().numpy()  # GAP 적용

            else:  # vit
                # ✅ Vision Transformer의 feature extraction 방식 수정
                outputs = image_model(inputs["pixel_values"])
                embedding = outputs.squeeze().numpy()

        # ✅ 임베딩 저장
        if embedding is not None:
            save_embedding(image_url, embedding)
            return embedding
        else:
            return None

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

        # ✅ 임베딩을 1차원으로 변환
        target_embedding = target_embedding.flatten()

        # 3. 비교할 이미지들 확인
        all_images = (
            db.query(ProductImage)
            .join(Product, ProductImage.product_id == Product.id)
            .filter(Product.category_id == 1)
            .filter(Product.id != product_id)
            .distinct(ProductImage.product_id)
            .all()
        )

        logger.info(f"Found {len(all_images)} images to compare")

        # 4. 유사도 계산
        product_similarities = {}
        for img in all_images:
            try:
                img_embedding = get_similar_image_embedding(img.url)
                if img_embedding is None:
                    continue

                # ✅ 임베딩을 1차원으로 변환
                img_embedding = img_embedding.flatten()

                similarity = float(
                    cosine_similarity([target_embedding], [img_embedding])[0][0]
                )

                if (
                    img.product_id not in product_similarities
                    or similarity > product_similarities[img.product_id]
                ):
                    product_similarities[img.product_id] = similarity

            except Exception as e:
                logger.error(
                    f"Error calculating similarity for image {img.url}: {str(e)}"
                )
                continue

        # 5. 결과 정렬
        sorted_similarities = [
            {"product_id": pid, "similarity": float(sim)}
            for pid, sim in sorted(
                product_similarities.items(), key=lambda x: x[1], reverse=True
            )
        ][:top_n]

        logger.info(f"Final results: {sorted_similarities}")
        return sorted_similarities

    except Exception as e:
        logger.error(f"Error in find_similar_images: {str(e)}")
        return []
