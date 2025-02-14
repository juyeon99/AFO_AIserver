from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.base_model import Product, Note, Bookmark, ProductImage, Spice
import torch
from services.mongo_service import MongoService
import logging

logger = logging.getLogger(__name__)

class PerfumeRecommender:
    def __init__(self, mongo_service: MongoService):
        # Sentence-BERT 모델 초기화
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # GPU 사용이 가능하면 GPU로, 아니면 CPU로 모델 이동
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MongoDB 서비스
        self.mongo_service = mongo_service
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        텍스트의 임베딩 벡터를 반환하는 메서드
        MongoDB에 저장된 임베딩이 있으면 사용하고, 없으면 새로 계산하여 저장
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        # MongoDB에서 먼저 확인
        cached_embedding = self.mongo_service.load_text_embedding(text)
        if cached_embedding is not None:
            return cached_embedding
            
        # 없으면 새로 계산하고 저장
        embedding = self.model.encode(text)
        self.mongo_service.save_text_embedding(text, embedding)
        return embedding
   
    def get_recommendations(self, member_id: str, db: Session, top_n: int = 5):
        """
        사용자가 북마크한 향수들의 공통 특성을 기반으로 새로운 향수를 추천
        
        Args:
            member_id (str): 사용자 ID
            db (Session): 데이터베이스 세션
            top_n (int): 추천할 향수 개수 (기본값: 5)
            
        Returns:
            list: 추천된 향수 정보 리스트. 유사도 점수 순으로 정렬됨
        """
        # 사용자가 북마크한 향수 목록 조회
        bookmarked_products = (
            db.query(Product)
            .join(Bookmark)
            .filter(Bookmark.member_id == member_id)
            .all()
        )
        
        if not bookmarked_products:
            return []
            
        # 북마크된 향수들의 공통 특성 추출
        common_features = self._extract_common_features(bookmarked_products, db)
        logger.info(f"추출된 공통 특성: {common_features}")
        
        # 공통 특성을 문자열로 변환하여 임베딩
        common_features_text = (
            f"Main accords: {', '.join(common_features['main_accords'])} "
            f"Spices: {', '.join(common_features['spices'])}"
        )
        target_embedding = self._get_embedding(common_features_text)
        
        # 추천 향수 찾기
        return self._find_similar_perfumes(
            target_embedding,
            common_features,
            bookmarked_products,
            db,
            top_n
        )
   
    def _extract_common_features(self, products, db):
        """
        북마크된 향수들의 공통 특성(향과 스파이스)을 추출
        
        Args:
            products (list): 향수 객체 리스트
            db (Session): 데이터베이스 세션
            
        Returns:
            dict: {
                'main_accords': [상위 2개 main_accord],
                'spices': [자주 등장한 스파이스 리스트]
            }
        """
        main_accords = {}
        spices = {}
        
        # 북마크한 향수 목록 로깅
        logger.info(f"\n=== 북마크한 향수 목록 ===")
        for product in products:
            logger.info(f"- {product.name_kr} ({product.brand}): {product.main_accord}")
        
        for product in products:
            # main_accord 빈도수 집계
            if product.main_accord in main_accords:
                main_accords[product.main_accord] += 1
            else:
                main_accords[product.main_accord] = 1
                
            # 해당 향수의 모든 스파이스 정보 조회
            notes_with_spices = (
                db.query(Note, Spice)
                .join(Spice, Note.spice_id == Spice.id)
                .filter(Note.product_id == product.id)
                .all()
            )
            
            # 스파이스 빈도수 집계
            for note, spice in notes_with_spices:
                if spice.name_kr in spices:
                    spices[spice.name_kr] += 1
                else:
                    spices[spice.name_kr] = 1

        # 빈도수에 따른 임계값 설정 (전체 북마크된 향수 중 30% 이상에서 등장한 향료 선택)
        threshold = len(products) * 0.3
        
        # main_accord 빈도수 로깅
        logger.info(f"\n=== Main Accord 빈도수 ===")
        for accord, count in sorted(main_accords.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {accord}: {count}회")
        
        # 스파이스 빈도수 로깅
        logger.info(f"\n=== 스파이스 빈도수 ===")
        for spice, count in sorted(spices.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {spice}: {count}회")
        
        result = {
            'main_accords': [k for k, v in sorted(main_accords.items(), key=lambda x: x[1], reverse=True)[:2]],
            'spices': [k for k, v in sorted(spices.items(), key=lambda x: x[1], reverse=True) 
                    if v >= threshold]
        }
        
        # 최종 선택된 공통 특성 로깅
        logger.info(f"\n=== 선택된 공통 특성 ===")
        logger.info(f"상위 Main Accords: {', '.join(result['main_accords'])}")
        logger.info(f"주요 스파이스 (빈도수 {threshold:.1f} 이상): {', '.join(result['spices'])}")
        logger.info("=====================\n")
        
        return result
   
    def _find_similar_perfumes(self, target_embedding, common_features, bookmarked_products, db, top_n):
        """
        주어진 특성과 유사한 향수를 찾아 추천
        
        Args:
            target_embedding (numpy.ndarray): 목표 특성의 임베딩 벡터
            common_features (dict): 공통 특성 정보
            bookmarked_products (list): 이미 북마크된 향수 리스트
            db (Session): 데이터베이스 세션
            top_n (int): 추천할 향수 개수
            
        Returns:
            list: 추천된 향수 정보 리스트 (유사도 점수 순으로 정렬됨)
        """
        bookmarked_ids = [p.id for p in bookmarked_products]
        
        # 모든 향수 데이터를 한 번에 조회 (북마크된 향수 제외)
        products_data = (
            db.query(Product, ProductImage.url, Note, Spice)
            .join(ProductImage, Product.id == ProductImage.product_id)
            .join(Note, Product.id == Note.product_id)
            .join(Spice, Note.spice_id == Spice.id)
            .filter(Product.id.notin_(bookmarked_ids))
            .all()
        )
        
        # 제품별로 데이터 그룹화
        grouped_products = {}
        for product, image_url, note, spice in products_data:
            if product.id not in grouped_products:
                grouped_products[product.id] = {
                    'product': product,
                    'image_url': image_url,
                    'spices': set()  # 중복 제거를 위해 set 사용
                }
            grouped_products[product.id]['spices'].add(spice.name_kr)
        
        # 임베딩 생성을 위한 텍스트 리스트와 제품 정보 리스트 준비
        texts = []
        product_info = []
        embeddings = []
        
        for product_data in grouped_products.values():
            product = product_data['product']
            spice_info = sorted(product_data['spices'])  # set을 정렬된 리스트로 변환
            
            # 각 제품의 특성을 텍스트로 변환
            text = f"Main accords: {product.main_accord} Spices: {', '.join(spice_info)}"
            embedding = self._get_embedding(text)
            
            texts.append(text)
            embeddings.append(embedding)
            product_info.append({
                "product_id": product.id,
                "name": product.name_kr,
                "brand": product.brand,
                "main_accord": product.main_accord,
                "image_url": product_data['image_url'],
                "spices": spice_info
            })
        
        if not embeddings:
            return []
            
        # 유사도 계산
        product_embeddings = np.array(embeddings)
        similarities = cosine_similarity([target_embedding], product_embeddings)[0]
        
        # 각 제품에 대한 추천 정보 생성
        candidates = []
        for i, (similarity, info) in enumerate(zip(similarities, product_info)):
            candidates.append({
                **info,
                "similarity_score": float(similarity),
                "common_features": {
                    "main_accord": info["main_accord"] in common_features["main_accords"],
                    "spices": [spice for spice in info["spices"] if spice in common_features["spices"]]
                }
            })
        
        # 유사도 점수가 높은 순으로 정렬하여 상위 n개 반환
        return sorted(candidates, key=lambda x: x["similarity_score"], reverse=True)[:top_n]