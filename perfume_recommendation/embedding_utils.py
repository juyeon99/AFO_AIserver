import numpy as np
from .services.mongo_service import MongoService

# MongoDB 서비스 인스턴스 생성
mongo_service = MongoService()

def save_embedding(image_url: str, embedding: np.ndarray):
    """MongoDB에 이미지 임베딩 저장"""
    return mongo_service.save_image_embedding(image_url, embedding)

def load_embedding(image_url: str):
    """MongoDB에서 이미지 임베딩 불러오기"""
    return mongo_service.load_image_embedding(image_url)

def save_text_embedding(text: str, embedding: np.ndarray):
    """MongoDB에 텍스트 임베딩 저장"""
    return mongo_service.save_text_embedding(text, embedding)

def load_text_embedding(text: str):
    """MongoDB에서 텍스트 임베딩 불러오기"""
    return mongo_service.load_text_embedding(text)