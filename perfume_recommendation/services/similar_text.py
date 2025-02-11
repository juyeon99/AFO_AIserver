from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .db_service import Product, Note
from perfume_recommendation.embedding_utils import save_text_embedding, load_text_embedding  # ✅ 캐시 추가

# ✅ 텍스트 모델 설정
TEXT_MODEL_TYPE = "mpnet"
TEXT_MODEL_CONFIG = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

text_model = SentenceTransformer(TEXT_MODEL_CONFIG[TEXT_MODEL_TYPE])


def get_similar_text_embedding(text: str):
    """텍스트 임베딩을 캐시에서 불러오거나, 새로 계산"""

    # ✅ 먼저 캐시에서 불러오기 시도
    cached_embedding = load_text_embedding(text)
    if cached_embedding is not None:
        print(f"✅ 캐시에서 텍스트 임베딩 불러옴: {text}")  # 🚀 디버깅 메시지 추가
        return cached_embedding  # 캐시가 있으면 바로 반환

    # 캐시에 없으면 새로 계산
    embedding = text_model.encode(text)

    # ✅ 새로 계산된 임베딩을 저장 (JSON 캐싱)
    save_text_embedding(text, embedding)
    return embedding


def find_similar_texts(product_id: int, db: Session, top_n: int = 5):
    """텍스트 기반 유사 향수 검색"""
    # 1. 기준 제품 정보 조회
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        return []

    # 2. 기준 제품의 노트 정보 조회
    notes = db.query(Note).filter(Note.product_id == product_id).all()
    note_info = " ".join([note.note_type for note in notes])

    # 3. 기준 제품의 메인 어코드와 설명 정보
    product_info = f"{product.main_accord} {product.content if product.content else ''}"

    # 4. 기준 제품의 임베딩 생성
    target_embedding = np.mean(
        [
            get_similar_text_embedding(note_info) * 2.0,  # 노트 정보 가중치
            get_similar_text_embedding(product.main_accord) * 1.5,  # 메인 어코드 가중치
            get_similar_text_embedding(product_info),  # 제품 설명
        ],
        axis=0,
    )

    # 5. 다른 향수 제품들과 유사도 계산
    similarities = []
    all_products = (
        db.query(Product)
        .filter(Product.category_id == 1, Product.id != product_id)
        .all()
    )

    for other_product in all_products:
        # 다른 제품의 노트 정보
        other_notes = db.query(Note).filter(Note.product_id == other_product.id).all()
        other_note_info = " ".join([note.note_type for note in other_notes])

        # 다른 제품의 메인 어코드와 설명 정보
        other_product_info = f"{other_product.main_accord} {other_product.content if other_product.content else ''}"

        # 다른 제품의 임베딩 생성
        other_embedding = np.mean(
            [
                get_similar_text_embedding(other_note_info) * 2.0,
                get_similar_text_embedding(other_product.main_accord) * 1.5,
                get_similar_text_embedding(other_product_info),
            ],
            axis=0,
        )

        # 유사도 계산
        similarity = cosine_similarity([target_embedding], [other_embedding])[0][0]
        similarities.append({"product_id": other_product.id, "similarity": similarity})

    # 6. 유사도 높은 순으로 정렬 후 상위 N개 선택
    sorted_similarities = sorted(
        similarities, key=lambda x: x["similarity"], reverse=True
    )[:top_n]
    return sorted_similarities
