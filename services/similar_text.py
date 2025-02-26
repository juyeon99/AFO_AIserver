import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import sessionmaker
from services.db_service import Product, Note, SessionLocal
from embedding_utils import save_text_embedding, load_text_embedding

# ✅ 텍스트 임베딩을 위한 모델 설정
# mpnet: Microsoft의 MPNet 모델 (성능이 좋지만 상대적으로 느림)
# minilm: 경량화된 BERT 모델 (빠르지만 성능은 약간 낮음)

# ✅ GPU 사용 가능하면 GPU로 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 텍스트 임베딩을 위한 모델 설정 (유지됨!)
TEXT_MODEL_TYPE = "mpnet"
TEXT_MODEL_CONFIG = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# ✅ 선택된 모델로 텍스트 임베딩 모델 초기화 + GPU 지원 추가
text_model = SentenceTransformer(TEXT_MODEL_CONFIG[TEXT_MODEL_TYPE]).to(device)

# ✅ 세션 팩토리를 생성하여 세션 객체를 만듦
Session = sessionmaker(bind=SessionLocal().bind)

def get_similar_text_embedding(text: str):
    """GPU 가속 적용"""
    if not text:
        text = ""

    cached_embedding = load_text_embedding(text)
    if cached_embedding is not None:
        return cached_embedding

    with torch.no_grad():
        embedding = text_model.encode(text, convert_to_tensor=True).cpu().numpy()  # ✅ GPU에서 연산 후 CPU로 변환

    save_text_embedding(text, embedding)
    return embedding

def find_similar_texts(product_id: int, top_n: int = 5):
    """텍스트 기반 유사 향수 추천 최적화"""
    
    # ✅ 새로운 DB 세션 생성
    db = SessionLocal()
    
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return []

        notes = db.query(Note.note_type).filter(Note.product_id == product_id).all()
        note_info = " ".join([n.note_type for n in notes]) if notes else ""

        embeddings_list = [
            get_similar_text_embedding(note_info) * 2.0,
            get_similar_text_embedding(product.main_accord or "") * 1.5,
            get_similar_text_embedding(product.content or ""),
        ]

        valid_embeddings = [emb for emb in embeddings_list if emb is not None]
        target_embedding = np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(768)

        all_products = db.query(Product).filter(Product.category_id == 1, Product.id != product_id).all()

        notes_dict = {p.id: [] for p in all_products}
        for note in db.query(Note.product_id, Note.note_type).filter(Note.product_id.in_([p.id for p in all_products])).all():
            notes_dict[note.product_id].append(note.note_type)

        embeddings = {
            p.id: np.mean([
                get_similar_text_embedding(" ".join(notes_dict[p.id])) * 2.0,
                get_similar_text_embedding(p.main_accord or "") * 1.5,
                get_similar_text_embedding(p.content or ""),
            ], axis=0)
            for p in all_products
        }

        target_embedding = target_embedding.reshape(1, -1)
        ids = list(embeddings.keys())
        vectors = np.array(list(embeddings.values()))

        similarities = cosine_similarity(target_embedding, vectors)[0]

        sorted_indices = np.argsort(similarities)[::-1][:top_n]
        return [{"product_id": ids[i], "similarity": float(similarities[i])} for i in sorted_indices]
    
    finally:
        db.close()  # ✅ 세션 종료