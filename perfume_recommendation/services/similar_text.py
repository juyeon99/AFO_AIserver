from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .db_service import Product, Note
from perfume_recommendation.embedding_utils import save_text_embedding, load_text_embedding  # ✅ 캐시 추가

# ✅ 텍스트 임베딩을 위한 모델 설정
# mpnet: Microsoft의 MPNet 모델 (성능이 좋지만 상대적으로 느림)
# minilm: 경량화된 BERT 모델 (빠르지만 성능은 약간 낮음)
TEXT_MODEL_TYPE = "mpnet"
# 각 모델의 Hugging Face 저장소 경로
TEXT_MODEL_CONFIG = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",     # 다국어 지원, SOTA 성능
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",     # 빠른 처리 속도
}

# 선택된 모델로 텍스트 임베딩 모델 초기화
text_model = SentenceTransformer(TEXT_MODEL_CONFIG[TEXT_MODEL_TYPE])


def get_similar_text_embedding(text: str):
    """
    텍스트의 임베딩 벡터를 생성하는 함수
    
    Args:
        text (str): 임베딩할 텍스트
        
    Returns:
        numpy.ndarray: 텍스트의 임베딩 벡터
        
    Note:
        - 캐시에서 먼저 임베딩을 찾아보고, 없으면 새로 생성
        - 생성된 임베딩은 자동으로 캐시에 저장됨
    """

    # ✅ 캐시에서 임베딩 확인
    cached_embedding = load_text_embedding(text)
    if cached_embedding is not None:
        print(f"✅ 캐시에서 텍스트 임베딩 불러옴: {text}")
        return cached_embedding  # 캐시가 있으면 바로 반환

    # ✅ 캐시에 없으면 새로 임베딩 생성
    embedding = text_model.encode(text)

    # ✅ 새로 생성된 임베딩을 캐시에 저장
    save_text_embedding(text, embedding)
    return embedding


def find_similar_texts(product_id: int, db: Session, top_n: int = 5):
    """
    텍스트 기반으로 유사한 향수를 찾는 함수
    
    Args:
        product_id (int): 기준이 되는 향수의 ID
        db (Session): 데이터베이스 세션
        top_n (int): 반환할 유사 향수의 개수
    
    Returns:
        list: 유사도가 높은 순으로 정렬된 향수 목록 [{product_id, similarity}, ...]
    """
    # ✅ 1. 기준 제품 정보 조회
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        return []

    # ✅ 2. 기준 제품의 노트 정보 조회 (top, middle, base notes)
    notes = db.query(Note).filter(Note.product_id == product_id).all()
    note_info = " ".join([note.note_type for note in notes])

    # ✅ 3. 기준 제품의 메인 어코드와 설명 정보 결합
    product_info = f"{product.main_accord} {product.content if product.content else ''}"

    # ✅ 4. 기준 제품의 임베딩 생성 (노트, 메인 어코드, 설명에 가중치 적용)
    target_embedding = np.mean(
        [
            get_similar_text_embedding(note_info) * 2.0,            # 노트 정보 (2배 가중치)
            get_similar_text_embedding(product.main_accord) * 1.5,  # 메인 어코드 (1.5배 가중치)
            get_similar_text_embedding(product_info),               # 제품 설명 (기본 가중치)
        ],
        axis=0,
    )

    # ✅ 5. 다른 향수 제품들과 유사도 계산
    similarities = []
    all_products = (
        db.query(Product)
        .filter(Product.category_id == 1, Product.id != product_id)  # 향수 카테고리만, 대상 제품 제외
        .all()
    )

    for other_product in all_products:
        # 다른 제품의 노트 정보
        other_notes = db.query(Note).filter(Note.product_id == other_product.id).all()
        other_note_info = " ".join([note.note_type for note in other_notes])

        # 다른 제품의 메인 어코드와 설명 정보
        other_product_info = f"{other_product.main_accord} {other_product.content if other_product.content else ''}"

        # 다른 제품의 임베딩 생성 (동일한 가중치 적용)
        other_embedding = np.mean(
            [
                get_similar_text_embedding(other_note_info) * 2.0,
                get_similar_text_embedding(other_product.main_accord) * 1.5,
                get_similar_text_embedding(other_product_info),
            ],
            axis=0,
        )

        # 코사인 유사도 계산
        similarity = cosine_similarity([target_embedding], [other_embedding])[0][0]
        similarities.append({"product_id": other_product.id, "similarity": similarity})

    # ✅ 6. 유사도 높은 순으로 정렬 후 상위 N개 선택
    sorted_similarities = sorted(
        similarities, key=lambda x: x["similarity"], reverse=True
    )[:top_n]
    return sorted_similarities
