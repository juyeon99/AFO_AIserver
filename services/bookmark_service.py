from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.base_model import Product, Note, Bookmark, ProductImage, Spice
import torch
from services.mongo_service import MongoService
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

class PerfumeRecommender:
    """향수 추천 시스템 클래스"""
    
    def __init__(self, mongo_service: MongoService):
        """초기화"""
        # 지연 로딩을 위한 변수들
        self._model = None         # 텍스트 임베딩 모델
        self.mongo_service = mongo_service  # MongoDB 연결
        self._embedding_dim = None  # 임베딩 벡터 차원

    @property # 메서드를 속성처럼 사용 가능하게 만드는 데코레이터
                # 호출 시점에 모델을 로드하는 지연 초기화(lazy initialization) 구현
                # self._model이 None일 때만 실제 모델을 로드하여 메모리 효율적 관리
    def model(self):
        """텍스트 임베딩 모델 로드"""
        if self._model is None:
            # 모델 최초 로드
            self._model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            # GPU 사용 가능시 GPU로 이동
            self._model = self._model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.eval()  # 추론 모드 설정
            
            # 임베딩 차원 확인
            dummy_text = "Test sentence for dimension check"
            dummy_embedding = self._model.encode(dummy_text)
            self._embedding_dim = dummy_embedding.shape[0]
            logger.info(f"모델 임베딩 차원: {self._embedding_dim}")
            
        return self._model

    def _get_threshold_values(self, product_count):
        """북마크 수에 따른 임계값 설정
        
        북마크 수가 적을수록 엄격한 기준을, 많을수록 느슨한 기준을 적용합니다.
        
        Args:
            product_count (int): 사용자가 북마크한 상품의 개수
            
        Returns:
            tuple: (최소_일치_항목_수, 유사도_임계값)
                - 첫 번째 값: 추천을 위해 필요한 최소 일치 항목 수
                - 두 번째 값: 항목 간 유사성을 판단하는 임계값(0~1 사이 값)
        """
        try:
            product_count = int(product_count)
            
            # 북마크 수에 따라 다른 임계값 반환
            if product_count <= 3:
                return 1, 0.5  # 적은 수의 북마크: 엄격한 기준 (최소 1개 항목 일치, 유사도 0.5 이상)
            elif product_count <= 6:
                return 2, 0.4  # 중간 수의 북마크 (최소 2개 항목 일치, 유사도 0.4 이상)
            elif product_count <= 10:
                return 3, 0.3  # 다수의 북마크 (최소 3개 항목 일치, 유사도 0.3 이상)
            else:
                return 4, 0.2  # 많은 수의 북마크: 느슨한 기준 (최소 4개 항목 일치, 유사도 0.2 이상)
                
        except Exception as e:
            logger.error(f"임계값 설정 오류: {str(e)}", exc_info=True)
            raise

    @lru_cache(maxsize=128) # LRU(Least Recently Used) 캐시 적용, 최근 사용된 128개의 결과를 메모리에 저장
    def _calculate_spice_diversity(self, product_spices_tuple, common_spices_tuple):
        """스파이스 다양성 점수 계산"""
        # 튜플을 리스트로 변환
        product_spices = list(product_spices_tuple) if isinstance(product_spices_tuple, tuple) else product_spices_tuple
        common_spices = list(common_spices_tuple) if isinstance(common_spices_tuple, tuple) else common_spices_tuple
        
        # 교집합 계산
        common = set(str(s) for s in product_spices) & set(str(s) for s in common_spices)
        
        # 다양성 점수 계산 (0.0 ~ 0.1)
        if common_spices:
            spice_overlap_ratio = len(common) / len(common_spices)
            return 0.1 * (1 - spice_overlap_ratio)
        return 0.0

    def _get_embedding(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 생성"""
        # 캐시된 임베딩 확인
        cached_embedding = self.mongo_service.load_text_embedding(text)
        
        if cached_embedding is not None:
            # 캐시된 임베딩 형식 및 차원 확인
            if not isinstance(cached_embedding, np.ndarray):
                cached_embedding = np.array(cached_embedding)
                
            if self._embedding_dim is not None and cached_embedding.shape[0] != self._embedding_dim:
                logger.warning("임베딩 차원 불일치로 재계산")
                cached_embedding = None
            else:
                return cached_embedding
            
        # 새로운 임베딩 생성
        embedding = self.model.encode(text)
        
        # 임베딩 캐시 저장
        try:
            self.mongo_service.save_text_embedding(text, embedding)
        except Exception as e:
            logger.warning(f"임베딩 캐싱 실패: {str(e)}")
            
        return embedding
    
    def _get_embeddings_batch(self, texts: list) -> list:
        """여러 텍스트 배치 임베딩 처리"""
        if not texts:
            return []
            
        # 캐시 확인 및 재계산 필요한 텍스트 식별
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.mongo_service.load_text_embedding(text)
            
            if cached_embedding is not None:
                # 캐시된 임베딩 검증
                if not isinstance(cached_embedding, np.ndarray):
                    cached_embedding = np.array(cached_embedding)
                
                if self._embedding_dim is not None and cached_embedding.shape[0] != self._embedding_dim:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
                    embeddings.append(None)
                else:
                    embeddings.append(cached_embedding)
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                embeddings.append(None)
        
        # 배치 처리로 새로운 임베딩 생성
        if texts_to_encode:
            batch_embeddings = self.model.encode(texts_to_encode, batch_size=32)
            
            # 결과 업데이트 및 캐시 저장
            for idx, embedding in zip(indices_to_encode, batch_embeddings):
                embeddings[idx] = embedding
                try:
                    self.mongo_service.save_text_embedding(texts[idx], embedding)
                except Exception as e:
                    logger.warning(f"임베딩 캐싱 실패 (인덱스 {idx}): {str(e)}")
        
        return embeddings

    def _extract_common_features_simple(self, products, product_spices):
        """북마크된 향수들의 공통 특성 추출"""
        try:
            # 향과 스파이스 빈도수 집계
            main_accords = {}
            spices = {}
            
            # 북마크 목록 로깅
            logger.info("\n=== 북마크한 향수 목록 ===")
            for product in products:
                logger.info(f"- {product.name_kr} ({product.brand}): {product.main_accord}")
            
            # 빈도수 계산
            for product in products:
                # main_accord 빈도수
                if product.main_accord in main_accords:
                    main_accords[product.main_accord] += 1
                else:
                    main_accords[product.main_accord] = 1
                    
                # 스파이스 빈도수
                if product.id in product_spices:
                    for spice_name in product_spices[product.id]:
                        if spice_name in spices:
                            spices[spice_name] += 1
                        else:
                            spices[spice_name] = 1

            # 임계값 설정
            product_count = len(products)
            accord_count, spice_threshold = self._get_threshold_values(product_count)
            accord_count = min(accord_count, len(set(p.main_accord for p in products)))
            threshold = product_count * spice_threshold
            
            # 빈도수 로깅
            logger.info("\n=== Main Accord 빈도수 ===")
            for accord, count in sorted(main_accords.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"- {accord}: {count}회")
            
            logger.info("\n=== 스파이스 빈도수 ===")
            for spice, count in sorted(spices.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"- {spice}: {count}회")
            
            # 최종 결과 생성
            result = {
                'main_accords': [k for k, v in sorted(main_accords.items(), key=lambda x: x[1], reverse=True)[:accord_count]],
                'spices': [k for k, v in sorted(spices.items(), key=lambda x: x[1], reverse=True) 
                        if float(v) >= float(threshold)]
            }
            
            # 결과 로깅
            logger.info("\n=== 선택된 공통 특성 ===")
            logger.info(f"상위 {accord_count}개 Main Accords: {', '.join(result['main_accords'])}")
            logger.info(f"주요 스파이스 (빈도수 {threshold:.1f}회({spice_threshold*100}%) 이상): {', '.join(result['spices'])}")
            logger.info("=====================\n")
            
            return result
            
        except Exception as e:
            logger.error(f"공통 특성 추출 오류: {str(e)}", exc_info=True)
            raise

    def _process_candidate_data_simple(self, candidates, images, notes_with_spices):
        """후보 향수 데이터 그룹화"""
        # 이미지 URL 그룹화
        product_images = {}
        for product_id, url in images:
            if product_id not in product_images:
                product_images[product_id] = []
            product_images[product_id].append(url)
        
        # 스파이스 정보 그룹화
        product_spices = {}
        for product_id, spice_name in notes_with_spices:
            if product_id not in product_spices:
                product_spices[product_id] = set()
            product_spices[product_id].add(spice_name)
        
        # 최종 데이터 구조화
        grouped_products = {}
        for product in candidates:
            grouped_products[product.id] = {
                'product': product,
                'image_urls': product_images.get(product.id, []),
                'spices': sorted(list(product_spices.get(product.id, set())))
            }
            
        return grouped_products

    def _find_similar_perfumes_simple(self, target_embedding, common_features, bookmarked_ids, grouped_products, top_n):
        """유사 향수 찾기"""
        try:
            start_time = time.time()
            
            if not grouped_products:
                return []
            
            # 후보 데이터 준비
            product_info = []
            texts = []
            
            # 텍스트 생성
            for product_id, data in grouped_products.items():
                product = data['product']
                spice_list = data['spices']
                
                text = f"Main accords: {product.main_accord} Spices: {', '.join(spice_list)}"
                texts.append(text)
                
                product_info.append({
                    "productId": product.id,
                    "nameKr": product.name_kr,
                    "brand": product.brand,
                    "mainAccord": product.main_accord,
                    "imageUrls": data['image_urls'],
                    "spices": spice_list
                })
            
            # 임베딩 계산
            logger.info(f"후보 향수 텍스트 수: {len(texts)}")
            all_embeddings = self._get_embeddings_batch(texts)
            
            embeddings_time = time.time()
            logger.info(f"임베딩 계산 시간: {embeddings_time - start_time:.2f}초")
                
            # 유효한 임베딩 필터링
            valid_embeddings = []
            valid_product_info = []
            
            for i, emb in enumerate(all_embeddings):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_product_info.append(product_info[i])
            
            logger.info(f"유효한 임베딩 수: {len(valid_embeddings)}/{len(all_embeddings)}")
            
            if not valid_embeddings:
                logger.error("유효한 임베딩이 없습니다.")
                return []
            
            try:
                # 임베딩 스택 생성
                product_embeddings = np.stack(valid_embeddings)
            except ValueError as e:
                # 차원 불일치 처리
                logger.error(f"임베딩 스택 오류: {str(e)}")
                logger.info("임베딩 차원 확인 중...")
                
                # 차원별 개수 확인
                dimension_counts = {}
                for i, emb in enumerate(valid_embeddings):
                    dim = emb.shape[0] if isinstance(emb, np.ndarray) else len(emb)
                    dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
                
                logger.info(f"차원별 임베딩 수: {dimension_counts}")
                
                if not dimension_counts:
                    return []
                
                # 가장 많은 차원으로 통일
                most_common_dim = max(dimension_counts, key=dimension_counts.get)
                logger.info(f"가장 많은 차원 {most_common_dim}로 통일")
                
                # 선택된 차원의 임베딩만 필터링
                filtered_embeddings = []
                filtered_product_info = []
                for i, emb in enumerate(valid_embeddings):
                    dim = emb.shape[0] if isinstance(emb, np.ndarray) else len(emb)
                    if dim == most_common_dim:
                        filtered_embeddings.append(emb)
                        filtered_product_info.append(valid_product_info[i])
                
                valid_embeddings = filtered_embeddings
                valid_product_info = filtered_product_info
                
                logger.info(f"필터링 후 임베딩 수: {len(valid_embeddings)}")
                
                if not valid_embeddings:
                    return []
                
                # 임베딩 스택 재시도
                product_embeddings = np.stack(valid_embeddings)
            
            # 타겟 임베딩 처리
            if not isinstance(target_embedding, np.ndarray):
                target_embedding = np.array(target_embedding)
            
            # 차원 불일치 확인 및 조정    
            if len(valid_embeddings) > 0 and target_embedding.shape[0] != valid_embeddings[0].shape[0]:
                logger.warning(f"타겟 임베딩 차원({target_embedding.shape[0]})이 " 
                           f"제품 임베딩 차원({valid_embeddings[0].shape[0]})과 일치하지 않습니다.")
                
                # 타겟 임베딩 재계산
                common_features_text = (
                    f"Main accords: {', '.join(common_features['main_accords'])} "
                    f"Spices: {', '.join(common_features['spices'])}"
                )
                target_embedding = self.model.encode(common_features_text)
                logger.info(f"타겟 임베딩 재계산 완료. 새 차원: {target_embedding.shape[0]}")
            
            # 차원 형태 맞추기
            if len(target_embedding.shape) == 1:
                target_embedding = target_embedding.reshape(1, -1)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(target_embedding, product_embeddings)[0]
            
            # 다양성 점수 계산 -> 기본 취향은 유지하면서 다양한 향수를 추천하기 위함
            # 1. main_accord 다양성
            main_accord_diversity = np.array([
                0.1 if str(info["mainAccord"]) not in [str(acc) for acc in common_features["main_accords"]] else 0.0
                for info in valid_product_info
            ])
            
            # 2. 스파이스 다양성 
            if common_features["spices"]:
                spice_diversity = np.array([
                    self._calculate_spice_diversity(tuple(info["spices"]), tuple(common_features["spices"]))
                    for info in valid_product_info
                ])
            else:
                spice_diversity = np.zeros(len(valid_product_info))
            
            # 최종 점수 계산 (유사도 75% + 다양성 25%)
            final_scores = (similarities * 0.75) + ((main_accord_diversity + spice_diversity) * 0.25)
            
            # 상위 N개 선정
            top_n = min(top_n, len(valid_product_info))
            if top_n == 0:
                return []
            
            top_indices = np.argsort(final_scores)[-top_n:][::-1]
            
            # 시간 측정
            similarity_time = time.time()
            logger.info(f"유사도 계산 시간: {similarity_time - embeddings_time:.2f}초")
            
            # 최종 결과 생성
            final_results = [
                {
                    "productId": valid_product_info[i]["productId"],
                    "nameKr": valid_product_info[i]["nameKr"],
                    "brand": valid_product_info[i]["brand"],
                    "mainAccord": valid_product_info[i]["mainAccord"],
                    "imageUrls": valid_product_info[i]["imageUrls"],
                    "spices": valid_product_info[i]["spices"]
                }
                for i in top_indices
            ]
            
            return final_results
            
        except Exception as e:
            logger.error(f"유사 향수 찾기 오류: {str(e)}", exc_info=True)
            logger.error(f"공통 특성: {common_features}")
            raise

    def get_recommendations(self, member_id: int, db: Session, top_n: int = 5):
        """향수 추천 메인 메서드"""
        start_time = time.time()
        try:
            # 1. 사용자 ID 검증
            member_id = int(member_id) if not isinstance(member_id, int) else member_id
            
            # 2. 북마크 향수 조회
            bookmarked_products = (
                db.query(Product)
                .join(Bookmark)
                .filter(Bookmark.member_id == member_id)
                .all()
            )
            
            if not bookmarked_products:
                logger.info(f"사용자 {member_id}의 북마크된 향수가 없습니다.")
                return []
            
            bookmarked_ids = [p.id for p in bookmarked_products]
            
            # 3. 북마크 향수의 스파이스 정보 조회
            bookmarked_notes_with_spices = (
                db.query(Note.product_id, Spice.name_kr)
                .join(Spice, Note.spice_id == Spice.id)
                .filter(Note.product_id.in_(bookmarked_ids))
                .all()
            )
            
            # 4. 스파이스 정보 그룹화
            bookmarked_spices = {}
            for product_id, spice_name in bookmarked_notes_with_spices:
                if product_id not in bookmarked_spices:
                    bookmarked_spices[product_id] = set()
                bookmarked_spices[product_id].add(spice_name)
            
            db_query_time = time.time()
            logger.info(f"북마크 데이터 쿼리 시간: {db_query_time - start_time:.2f}초")
            
            # 5. 공통 특성 추출
            common_features = self._extract_common_features_simple(
                bookmarked_products, 
                bookmarked_spices
            )
            logger.info(f"추출된 공통 특성: {common_features}")
            
            # 6. 공통 특성 텍스트화
            common_features_text = (
                f"Main accords: {', '.join(common_features['main_accords'])} "
                f"Spices: {', '.join(common_features['spices'])}"
            )
            
            # 7. 타겟 임베딩 계산
            target_embedding = self._get_embedding(common_features_text)
            
            # 8. 후보 향수 데이터 병렬 조회
            with ThreadPoolExecutor(max_workers=1) as executor:
                def get_candidate_products_data(session_factory, bookmarked_ids):
                    """후보 향수 데이터 조회"""
                    session = session_factory()
                    try:
                        # 8-1. 북마크 제외 향수 조회
                        candidates = (
                            session.query(Product)
                            .filter(Product.id.notin_(bookmarked_ids))
                            .all()
                        )
                        candidate_ids = [p.id for p in candidates]
                        
                        # 8-2. 이미지 URL 조회
                        images = (
                            session.query(ProductImage.product_id, ProductImage.url)
                            .filter(ProductImage.product_id.in_(candidate_ids))
                            .all()
                        )
                        
                        # 8-3. 스파이스 정보 조회
                        notes_with_spices = (
                            session.query(Note.product_id, Spice.name_kr)
                            .join(Spice, Note.spice_id == Spice.id)
                            .filter(Note.product_id.in_(candidate_ids))
                            .all()
                        )
                        
                        return {
                            'candidates': candidates,
                            'images': images,
                            'notes_with_spices': notes_with_spices
                        }
                    finally:
                        session.close()
                
                session_factory = lambda: Session(bind=db.get_bind())
                
                # 9. 병렬 처리 실행
                candidates_future = executor.submit(
                    get_candidate_products_data,
                    session_factory,
                    bookmarked_ids
                )
                
                candidate_data = candidates_future.result()
            
            parallel_time = time.time()
            logger.info(f"병렬 처리 시간: {parallel_time - db_query_time:.2f}초")
            
            # 10. 후보 데이터 정리
            grouped_products = self._process_candidate_data_simple(
                candidate_data['candidates'],
                candidate_data['images'],
                candidate_data['notes_with_spices']
            )
            
            processing_time = time.time()
            logger.info(f"데이터 가공 시간: {processing_time - parallel_time:.2f}초")
            
            # 11. 유사도 기반 추천
            recommendations = self._find_similar_perfumes_simple(
                target_embedding,
                common_features,
                bookmarked_ids,
                grouped_products,
                top_n
            )
            
            # 실행 시간 측정
            end_time = time.time()
            logger.info(f"전체 추천 시간: {end_time - start_time:.2f}초")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"추천 처리 중 오류 발생: {str(e)}", exc_info=True)
            raise