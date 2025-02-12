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

# ✅ 이미지 처리를 위한 모델 선택
# 'convnext': Meta AI의 최신 CNN 모델 (Hugging Face)
# 'swin': torchvision의 Swin Transformer V2 모델 (이미지 분류 및 특징 추출 최적화)
# 'vit': torchvision의 Vision Transformer 모델 (Transformer 기반 이미지 인식)
IMAGE_MODEL_TYPE = "convnext"  # 'convnext', 'swin', 'vit' 중 하나

# ✅ 모델 및 전처리기 초기화
if IMAGE_MODEL_TYPE == "convnext":
    # ConvNext 모델 설정 (Hugging Face에서 제공)
    model_path = "facebook/convnext-base-224"
    image_model = ConvNextModel.from_pretrained(model_path)
    image_processor = ConvNextImageProcessor.from_pretrained(model_path)
elif IMAGE_MODEL_TYPE == "swin":
    # Swin Transformer V2 모델 설정 (torchvision 제공)
    weights = Swin_V2_B_Weights.IMAGENET1K_V1   # ImageNet으로 학습된 가중치
    image_model = swin_v2_b(weights=weights)    # 모델 생성
    image_processor = weights.transforms()      # 이미지 전처리 파이프라인
else:  # vit
    image_model = vit_b_16(pretrained=True)
    image_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224")

image_model.eval()      # 모델을 평가 모드로 설정 (학습 비활성화)


def get_similar_image_embedding(image_url: str):
    """
    이미지 URL로부터 임베딩 벡터를 생성하는 함수
    
    Args:
        image_url (str): 이미지의 URL 주소
    
    Returns:
        numpy.ndarray: 이미지의 임베딩 벡터. 실패 시 None 반환
    """

    # ✅ 캐시된 임베딩이 있는지 확인
    cached_embedding = load_embedding(image_url)
    if cached_embedding is not None:
        print(f"✅ 캐시에서 임베딩 불러옴: {image_url}")
        return cached_embedding  # 캐시가 있으면 바로 반환

    try:
        # ✅ 이미지 다운로드 및 전처리
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        # ✅ 모델별 이미지 처리 및 임베딩 생성
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            if IMAGE_MODEL_TYPE == "convnext":
                # ConvNext 모델의 특징 추출
                outputs = image_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            elif IMAGE_MODEL_TYPE == "swin":
                # Swin V2의 특징 추출
                features = image_model.forward_features(inputs["pixel_values"])
                embedding = features.mean(dim=[2, 3]).squeeze().numpy()  # GAP 적용

            else:  # vit
                # Vision Transformer의 특징 추출
                outputs = image_model(inputs["pixel_values"])
                embedding = outputs.squeeze().numpy()

        # ✅ 임베딩 저장 및 반환
        if embedding is not None:
            save_embedding(image_url, embedding)    # MongoDB에 임베딩 저장
            return embedding
        else:
            return None

    except Exception:
        return None


def find_similar_images(product_id: int, db: Session, top_n: int = 5):
    """
    이미지 기반으로 유사한 향수를 찾는 함수
    
    Args:
        product_id (int): 기준이 되는 향수의 ID
        db (Session): 데이터베이스 세션
        top_n (int): 반환할 유사 향수의 개수
    
    Returns:
        list: 유사도가 높은 순으로 정렬된 향수 목록 [{product_id, similarity}, ...]
    """
    
    try:
        # ✅ 대상 향수의 이미지 가져오기
        target_image = (
            db.query(ProductImage).filter(ProductImage.product_id == product_id).first()
        )
        if not target_image:
            return []

        # ✅ 대상 이미지의 임베딩 생성
        target_embedding = get_similar_image_embedding(target_image.url)
        if target_embedding is None:
            return []

        # ✅ 임베딩을 1차원으로 변환 (유사도 계산을 위해)
        target_embedding = target_embedding.flatten()

        # ✅ 비교할 향수 이미지들 가져오기
        all_images = (
            db.query(ProductImage)
            .join(Product, ProductImage.product_id == Product.id)
            .filter(Product.category_id == 1)
            .filter(Product.id != product_id)   # 대상 향수 제외
            .distinct(ProductImage.product_id)  # 향수당 하나의 이미지만
            .all()
        )

        logger.info(f"Found {len(all_images)} images to compare")

        # ✅ 각 향수와의 유사도 계산
        product_similarities = {}
        for img in all_images:
            try:
                img_embedding = get_similar_image_embedding(img.url)
                if img_embedding is None:
                    continue

                # 임베딩을 1차원으로 변환
                img_embedding = img_embedding.flatten()

                # 코사인 유사도 계산 (0~1 사이 값)
                similarity = float(
                    cosine_similarity([target_embedding], [img_embedding])[0][0]
                )

                # 더 높은 유사도로 업데이트
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

        # ✅ 유사도 기준으로 상위 N개 선택
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
