from sqlalchemy.orm import scoped_session, sessionmaker, Session
import torch
from torchvision.models import vit_b_16, swin_v2_b, Swin_V2_B_Weights
from transformers import ConvNextModel, ConvNextImageProcessor
from sklearn.metrics.pairwise import cosine_similarity
from services.db_service import Product, ProductImage, SessionLocal
from embedding_utils import save_embedding, load_embedding
import logging
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# ✅ 이미지 처리를 위한 모델 선택
# 'convnext': Meta AI의 최신 CNN 모델 (Hugging Face)
# 'swin': torchvision의 Swin Transformer V2 모델 (이미지 분류 및 특징 추출 최적화)
# 'vit': torchvision의 Vision Transformer 모델 (Transformer 기반 이미지 인식)

# ✅ GPU 사용 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 사용할 이미지 모델 타입 설정
IMAGE_MODEL_TYPE = "convnext"

# ✅ 선택된 모델 타입에 따른 모델과 전처리기 초기화
if IMAGE_MODEL_TYPE == "convnext":
    # ConvNext 모델 설정 (Hugging Face에서 제공)
    model_path = "facebook/convnext-base-224"
    image_model = ConvNextModel.from_pretrained(model_path).to(device)
    image_processor = ConvNextImageProcessor.from_pretrained(model_path)
elif IMAGE_MODEL_TYPE == "swin":
    # Swin Transformer V2 모델 설정 (torchvision 제공)
    weights = Swin_V2_B_Weights.IMAGENET1K_V1  # ImageNet으로 학습된 가중치
    image_model = swin_v2_b(weights=weights).to(device)
    image_processor = weights.transforms()  # 이미지 전처리 파이프라인
else:  # vit
    # Vision Transformer 모델 설정 (torchvision 제공)
    image_model = vit_b_16(pretrained=True).to(device)
    image_processor = ConvNextImageProcessor.from_pretrained(
        "facebook/convnext-base-224"
    )

# ✅ 모델을 평가 모드로 설정 (학습 비활성화)
image_model.eval()

# ✅ 멀티스레딩을 위한 스레드 풀 생성
executor = ThreadPoolExecutor(max_workers=4)


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
        return cached_embedding

    try:
        # ✅ 이미지 다운로드 후 변환 필요
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt").to(device)
            outputs = image_model(**inputs)

            # ✅ 차원 변환 추가 (1D → 2D 변환)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embedding = np.array(embedding).reshape(1, -1)  # ✅ 차원 변환 추가!

        save_embedding(image_url, embedding)
        return embedding
    except Exception as e:
        logger.error(f"Error processing image {image_url}: {e}")
        return None
    
# ✅ 멀티스레딩 환경에서 SQLAlchemy 세션 충돌을 방지하는 스레드별 세션 팩토리 생성
thread_local_session = scoped_session(sessionmaker(bind=SessionLocal().bind))

def find_similar_images(product_id: int, top_n: int = 5):
    """이미지 기반 유사 향수 추천"""

    # ✅ 새로운 DB 세션 생성
    db = SessionLocal()

    try:
        # ✅ 대상 향수의 이미지 가져오기
        target_image = db.query(ProductImage).filter(ProductImage.product_id == product_id).first()
        if not target_image:
            return []

        target_embedding = get_similar_image_embedding(target_image.url)
        if target_embedding is None:
            return []

        all_images = (
            db.query(ProductImage, Product)
            .join(Product, ProductImage.product_id == Product.id)
            .filter(Product.category_id == 1)
            .filter(Product.id != product_id)
            .distinct(ProductImage.product_id)
            .all()
        )

        def process_image(img, target_embedding):
            img_embedding = get_similar_image_embedding(img.ProductImage.url)
            if img_embedding is None:
                return None

            # ✅ 차원 변환 (1D → 2D 변환)
            target_embedding = np.array(target_embedding).reshape(1, -1)  # 1D → 2D
            img_embedding = np.array(img_embedding).reshape(1, -1)  # 1D → 2D

            # ✅ 유사도 계산
            similarity = cosine_similarity(target_embedding, img_embedding)[0][0]
            return img.Product.id, similarity

        results = list(executor.map(lambda img: process_image(img, target_embedding), all_images))
        results = [{"product_id": pid, "similarity": sim} for r in results if r is not None for pid, sim in [r]]

        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_n]

    finally:
        db.close()

