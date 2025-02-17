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
   
    def _get_threshold_values(self, product_count):
        """
        북마크 수에 따른 main accord 개수와 스파이스 임계값을 반환

        Args:
            product_count (int): 북마크된 향수 개수

        Returns:
            tuple: (accord_count, spice_threshold)
                - accord_count (int): 선택할 main accord 개수
                - spice_threshold (float): 스파이스 선택 임계 비율
        """
        product_count = int(product_count)
        if product_count <= 3:
            return 1, 0.5  # 1개 선택, 50% 이상 등장
        elif product_count <= 6:
            return 2, 0.4  # 2개 선택, 40% 이상 등장
        elif product_count <= 10:
            return 3, 0.3  # 3개 선택, 30% 이상 등장
        else:
            return 4, 0.2  # 4개 선택, 20% 이상 등장
        

    def _extract_common_features(self, products, db):
        """
        북마크된 향수들의 공통 특성(향과 스파이스)을 추출
        
        Args:
            products (list): 향수 객체 리스트
            db (Session): 데이터베이스 세션
            
        Returns:
            dict: {
                'main_accords': [상위 N개 main_accord],
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

        # 북마크 수에 따른 임계값 설정
        product_count = len(products)
        accord_count, spice_threshold = self._get_threshold_values(product_count)
        
        # accord_count 최대값 제한
        accord_count = min(accord_count, len(set(p.main_accord for p in products)))
        threshold = product_count * spice_threshold
        
        # main_accord 빈도수 로깅
        logger.info(f"\n=== Main Accord 빈도수 ===")
        for accord, count in sorted(main_accords.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {accord}: {count}회")
        
        # 스파이스 빈도수 로깅
        logger.info(f"\n=== 스파이스 빈도수 ===")
        for spice, count in sorted(spices.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {spice}: {count}회")
        
        result = {
            'main_accords': [k for k, v in sorted(main_accords.items(), key=lambda x: x[1], reverse=True)[:accord_count]],
            'spices': [k for k, v in sorted(spices.items(), key=lambda x: x[1], reverse=True) 
                    if v >= threshold]
        }
        
        # 최종 선택된 공통 특성 로깅
        logger.info(f"\n=== 선택된 공통 특성 ===")
        logger.info(f"상위 {accord_count}개 Main Accords: {', '.join(result['main_accords'])}")
        logger.info(f"주요 스파이스 (빈도수 {threshold:.1f}회({spice_threshold*100}%) 이상): {', '.join(result['spices'])}")
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
            list: 추천된 향수 정보 리스트 (최종 점수 순으로 정렬됨)
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
            # 제품 정보 조회
            product = product_data['product']
            spice_info = sorted(product_data['spices'])  # set을 정렬된 리스트로 변환
            
            # 각 제품의 특성을 텍스트로 변환
            text = f"Main accords: {product.main_accord} Spices: {', '.join(spice_info)}"
            
            # 임베딩 생성
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
            
        # embeddings 리스트를 NumPy 배열로 변환
        product_embeddings = np.array(embeddings)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity([target_embedding], product_embeddings)[0]
        
        # 각 제품에 대한 추천 정보 생성
        candidates = []
        for i, (similarity, info) in enumerate(zip(similarities, product_info)):
            # 해당 향수의 특성이 얼마나 다양한지 계산
            diversity_score = 0
            
            # 디버깅을 위한 로그 추가
            logger.info(f"main_accord type: {type(info['main_accord'])}")
            logger.info(f"main_accords type: {type(common_features['main_accords'])}")
            logger.info(f"main_accord value: {info['main_accord']}")
            logger.info(f"main_accords value: {common_features['main_accords']}")

            # 타입 체크 추가
            if info["main_accord"] is not None and info["main_accord"] not in common_features["main_accords"]:
                diversity_score += 0.1
                        
            # 공통된 spices의 비율이 낮을수록 다양성 점수 증가
            common_spices = set(info["spices"]) & set(common_features["spices"])
            if common_spices:
                spice_overlap_ratio = len(common_spices) / len(common_features["spices"])
                diversity_score += 0.1 * (1 - spice_overlap_ratio)
            
            # 최종 점수는 유사도(75%)와 다양성(25%)의 가중 평균
            final_score = (similarity * 0.75) + (diversity_score * 0.25)
            
            candidates.append({
            **info,
            "similarity_score": float(similarity),
            "diversity_score": float(diversity_score),
            "final_score": float(final_score),
            "common_features": {
                "main_accord": str(info["main_accord"]) in [str(x) for x in common_features["main_accords"]],
                "spices": [spice for spice in info["spices"] if str(spice) in [str(x) for x in common_features["spices"]]]
                }
            })
        
        # 최종 점수로 정렬하여 상위 n개 반환
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_n]