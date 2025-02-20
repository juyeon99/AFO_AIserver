import numpy as np
from scentlens_mongo_service import ScentlensMongoService

# MongoDB 서비스 인스턴스 생성
mongo_service = ScentlensMongoService()

def save_jina_image_embedding(id: int, image_url: str, product_id: int, embedding: np.ndarray):
    """MongoDB에 jina 이미지 임베딩 저장"""
    return mongo_service.save_jina_image_embedding(id, image_url, product_id, embedding)

def load_jina_image_embedding(image_url: str):
    """MongoDB에서 jina 이미지 임베딩 불러오기"""
    return mongo_service.load_jina_image_embedding(image_url)

def delete_jina_image_embedding(image_url: str):
    """MongoDB에서 특정 URL의 이미지 임베딩 삭제"""
    return mongo_service.delete_jina_image_embedding(image_url)

def clear_jina_image_embeddings():
    """MongoDB에서 모든 이미지 임베딩 삭제 (컬렉션 비우기)"""
    return mongo_service.clear_jina_image_embeddings()