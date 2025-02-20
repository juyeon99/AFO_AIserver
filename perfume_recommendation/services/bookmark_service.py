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
    def __init__(self, mongo_service: MongoService):
        # 지연 로딩을 위한 초기화
        self._model = None
        self.mongo_service = mongo_service
        self._embedding_dim = None  # 임베딩 차원 저장
    
    @property
    def model(self):
        """모델 지연 로딩 (원래 모델 유지)"""
        if self._model is None:
            # 원래 사용하던 모델 유지
            self._model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self._model = self._model.to('cuda' if torch.cuda.is_available() else 'cpu')
            # 추론 모드로 설정하여 성능 최적화
            self._model.eval()
            
            # 임베딩 차원 기록
            dummy_text = "Test sentence for dimension check"
            dummy_embedding = self._model.encode(dummy_text)
            self._embedding_dim = dummy_embedding.shape[0]
            logger.info(f"모델 임베딩 차원: {self._embedding_dim}")
            
        return self._model
    
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
        
        # 캐시된 임베딩이 있고 차원이 현재 모델과 일치하는지 확인
        if cached_embedding is not None:
            # 캐시된 임베딩을 numpy 배열로 변환
            if not isinstance(cached_embedding, np.ndarray):
                cached_embedding = np.array(cached_embedding)
                
            # 차원 확인
            if self._embedding_dim is not None and cached_embedding.shape[0] != self._embedding_dim:
                logger.warning(f"캐시된 임베딩 차원({cached_embedding.shape[0]})이 모델 차원({self._embedding_dim})과 일치하지 않아 재계산합니다.")
                cached_embedding = None
            else:
                return cached_embedding
            
        # 없거나 차원이 맞지 않으면 새로 계산
        embedding = self.model.encode(text)
        
        # 백그라운드 저장 없이 즉시 저장 (스레드 안전)
        try:
            self.mongo_service.save_text_embedding(text, embedding)
        except Exception as e:
            # 캐시 저장 실패해도 계속 진행
            logger.warning(f"임베딩 캐싱 실패: {str(e)}")
            
        return embedding
    
    def _get_embeddings_batch(self, texts: list) -> list:
        """여러 텍스트의 임베딩을 배치로 처리 - 차원 일관성 보장"""
        if not texts:
            return []
            
        # 캐시 확인
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.mongo_service.load_text_embedding(text)
            
            if cached_embedding is not None:
                # 차원 확인
                if not isinstance(cached_embedding, np.ndarray):
                    cached_embedding = np.array(cached_embedding)
                
                if self._embedding_dim is not None and cached_embedding.shape[0] != self._embedding_dim:
                    # 차원이 맞지 않으면 재계산 대상에 추가
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
                    embeddings.append(None)  # 임시 None
                else:
                    embeddings.append(cached_embedding)
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                embeddings.append(None)  # 임시 None
        
        # 배치로 인코딩
        if texts_to_encode:
            batch_embeddings = self.model.encode(texts_to_encode, batch_size=32)
            
            # 결과 업데이트
            for idx, embedding in zip(indices_to_encode, batch_embeddings):
                embeddings[idx] = embedding
                
                # 캐시 업데이트
                try:
                    self.mongo_service.save_text_embedding(texts[idx], embedding)
                except Exception as e:
                    logger.warning(f"임베딩 캐싱 실패 (인덱스 {idx}): {str(e)}")
        
        return embeddings
   
    def get_recommendations(self, member_id: int, db: Session, top_n: int = 5):
        """
        사용자가 북마크한 향수들의 공통 특성을 기반으로 새로운 향수를 추천
        
        Args:
            member_id (int): 사용자 ID
            db (Session): 데이터베이스 세션
            top_n (int): 추천할 향수 개수 (기본값: 5)
            
        Returns:
            list: 추천된 향수 정보 리스트. 유사도 점수 순으로 정렬됨
        """
        start_time = time.time()
        try:
            # 타입 안전을 위한 명시적 형변환
            member_id = int(member_id) if not isinstance(member_id, int) else member_id
            
            # 사용자가 북마크한 향수 목록 조회 (joinedload 없이)
            bookmarked_products = (
                db.query(Product)
                .join(Bookmark)
                .filter(Bookmark.member_id == member_id)
                .all()
            )
            
            if not bookmarked_products:
                logger.info(f"사용자 {member_id}의 북마크된 향수가 없습니다.")
                return []
            
            # 북마크된 향수 ID 추출
            bookmarked_ids = [p.id for p in bookmarked_products]
            
            # 북마크된 향수들의 노트와 스파이스 정보 별도 조회
            bookmarked_notes_with_spices = (
                db.query(Note.product_id, Spice.name_kr)
                .join(Spice, Note.spice_id == Spice.id)
                .filter(Note.product_id.in_(bookmarked_ids))
                .all()
            )
            
            # 제품별 스파이스 그룹화
            bookmarked_spices = {}
            for product_id, spice_name in bookmarked_notes_with_spices:
                if product_id not in bookmarked_spices:
                    bookmarked_spices[product_id] = set()
                bookmarked_spices[product_id].add(spice_name)
            
            db_query_time = time.time()
            logger.info(f"북마크 데이터 쿼리 시간: {db_query_time - start_time:.2f}초")
                
            # 북마크된 향수들의 공통 특성 추출 - 스파이스 정보를 별도로 전달
            common_features = self._extract_common_features_simple(
                bookmarked_products, 
                bookmarked_spices
            )
            logger.info(f"추출된 공통 특성: {common_features}")
            
            # 공통 특성을 문자열로 변환
            common_features_text = (
                f"Main accords: {', '.join(common_features['main_accords'])} "
                f"Spices: {', '.join(common_features['spices'])}"
            )
            
            # 타겟 임베딩 계산 (asyncio 오류 방지를 위해 메인 스레드에서 실행)
            target_embedding = self._get_embedding(common_features_text)
            
            # 후보 향수 데이터만 병렬로 조회
            with ThreadPoolExecutor(max_workers=1) as executor:
                # 후보 데이터 조회 함수
                def get_candidate_products_data(session_factory, bookmarked_ids):
                    # 새로운 세션 생성 (스레드 안전성)
                    session = session_factory()
                    try:
                        # 후보 향수 조회
                        candidates = (
                            session.query(Product)
                            .filter(Product.id.notin_(bookmarked_ids))
                            .all()
                        )
                        candidate_ids = [p.id for p in candidates]
                        
                        # 후보 향수의 이미지 URL 조회
                        images = (
                            session.query(ProductImage.product_id, ProductImage.url)
                            .filter(ProductImage.product_id.in_(candidate_ids))
                            .all()
                        )
                        
                        # 후보 향수의 노트와 스파이스 조회
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
                
                # 새로운 세션을 생성하는 함수
                session_factory = lambda: Session(bind=db.get_bind())
                
                # 병렬 실행
                candidates_future = executor.submit(
                    get_candidate_products_data,
                    session_factory,
                    bookmarked_ids
                )
                
                # 결과 가져오기
                candidate_data = candidates_future.result()
            
            parallel_time = time.time()
            logger.info(f"병렬 처리 시간: {parallel_time - db_query_time:.2f}초")
            
            # 후보 데이터 가공 - 제품별로 이미지와 스파이스 정보 그룹화
            grouped_products = self._process_candidate_data_simple(
                candidate_data['candidates'],
                candidate_data['images'],
                candidate_data['notes_with_spices']
            )
            
            processing_time = time.time()
            logger.info(f"데이터 가공 시간: {processing_time - parallel_time:.2f}초")
            
            # 추천 향수 찾기
            recommendations = self._find_similar_perfumes_simple(
                target_embedding,
                common_features,
                bookmarked_ids,
                grouped_products,
                top_n
            )
            
            end_time = time.time()
            logger.info(f"전체 추천 시간: {end_time - start_time:.2f}초")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"get_recommendations 내부 예외: {str(e)}", exc_info=True)
            raise  # 상위로 예외 전파
   
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
        try:
            product_count = int(product_count)
            
            if product_count <= 3:
                return 1, 0.5  # 1개 선택, 50% 이상 등장
            elif product_count <= 6:
                return 2, 0.4  # 2개 선택, 40% 이상 등장
            elif product_count <= 10:
                return 3, 0.3  # 3개 선택, 30% 이상 등장
            else:
                return 4, 0.2  # 4개 선택, 20% 이상 등장
        except Exception as e:
            logger.error(f"_get_threshold_values 예외: {str(e)}", exc_info=True)
            raise
        
    def _extract_common_features_simple(self, products, product_spices):
        """
        북마크된 향수들의 공통 특성(향과 스파이스)을 추출 - 관계 없이 작동하는 버전
        
        Args:
            products (list): 향수 객체 리스트
            product_spices (dict): {product_id: set(spice_names)} 형태의 스파이스 정보
            
        Returns:
            dict: {
                'main_accords': [상위 N개 main_accord],
                'spices': [자주 등장한 스파이스 리스트]
            }
        """
        try:
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
                    
                # 해당 제품의 스파이스 정보 사용
                if product.id in product_spices:
                    for spice_name in product_spices[product.id]:
                        if spice_name in spices:
                            spices[spice_name] += 1
                        else:
                            spices[spice_name] = 1

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
                        if float(v) >= float(threshold)]
            }
            
            # 최종 선택된 공통 특성 로깅
            logger.info(f"\n=== 선택된 공통 특성 ===")
            logger.info(f"상위 {accord_count}개 Main Accords: {', '.join(result['main_accords'])}")
            logger.info(f"주요 스파이스 (빈도수 {threshold:.1f}회({spice_threshold*100}%) 이상): {', '.join(result['spices'])}")
            logger.info("=====================\n")
            
            return result
        except Exception as e:
            logger.error(f"_extract_common_features_simple 예외: {str(e)}", exc_info=True)
            raise
            
    def _process_candidate_data_simple(self, candidates, images, notes_with_spices):
        """
        후보 향수 데이터를 제품별로 그룹화
        
        Args:
            candidates (list): 후보 향수 객체 리스트
            images (list): (product_id, url) 튜플 리스트
            notes_with_spices (list): (product_id, spice_name) 튜플 리스트
            
        Returns:
            dict: 제품별로 그룹화된 데이터
        """
        # 이미지 URL을 제품 ID별로 그룹화
        product_images = {}
        for product_id, url in images:
            if product_id not in product_images:
                product_images[product_id] = []
            product_images[product_id].append(url)
        
        # 스파이스를 제품 ID별로 그룹화
        product_spices = {}
        for product_id, spice_name in notes_with_spices:
            if product_id not in product_spices:
                product_spices[product_id] = set()
            product_spices[product_id].add(spice_name)
        
        # 최종 그룹화된 데이터 생성
        grouped_products = {}
        for product in candidates:
            grouped_products[product.id] = {
                'product': product,
                'image_urls': product_images.get(product.id, []),
                'spices': sorted(list(product_spices.get(product.id, set())))
            }
            
        return grouped_products
   
    def _find_similar_perfumes_simple(self, target_embedding, common_features, bookmarked_ids, grouped_products, top_n):
        """
        주어진 특성과 유사한 향수를 찾아 추천 - 차원 확인 로직 추가
        """
        try:
            start_time = time.time()
            
            if not grouped_products:
                return []
            
            # 임베딩 및 제품 정보 준비
            product_info = []
            texts = []
            
            # 텍스트 준비
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
            
            # 배치 처리로 임베딩 계산
            logger.info(f"후보 향수 텍스트 수: {len(texts)}")
            all_embeddings = self._get_embeddings_batch(texts)
            
            embeddings_time = time.time()
            logger.info(f"임베딩 계산 시간: {embeddings_time - start_time:.2f}초")
                
            # 유효한 임베딩과 해당 제품 정보 필터링
            valid_embeddings = []
            valid_product_info = []
            
            for i, emb in enumerate(all_embeddings):
                if emb is None:
                    continue
                    
                valid_embeddings.append(emb)
                valid_product_info.append(product_info[i])
            
            logger.info(f"유효한 임베딩 수: {len(valid_embeddings)}/{len(all_embeddings)}")
            
            if not valid_embeddings:
                logger.error("유효한 임베딩이 없습니다.")
                return []
            
            # 유효한 임베딩들을 NumPy 배열로 변환
            try:
                product_embeddings = np.stack(valid_embeddings)
            except ValueError as e:
                # 차원이 맞지 않는 경우 처리
                logger.error(f"임베딩 스택 중 오류: {str(e)}")
                
                # 모든 임베딩 차원 확인
                logger.info("임베딩 차원 확인 중...")
                dimension_counts = {}
                for i, emb in enumerate(valid_embeddings):
                    dim = emb.shape[0] if isinstance(emb, np.ndarray) else len(emb)
                    if dim not in dimension_counts:
                        dimension_counts[dim] = 0
                    dimension_counts[dim] += 1
                
                logger.info(f"차원별 임베딩 수: {dimension_counts}")
                
                # 가장 많은 차원 선택
                if not dimension_counts:
                    return []
                    
                most_common_dim = max(dimension_counts, key=dimension_counts.get)
                logger.info(f"가장 많은 차원 {most_common_dim}로 통일합니다.")
                
                # 선택한 차원으로 필터링
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
                    
                # 다시 스택 시도
                product_embeddings = np.stack(valid_embeddings)
            
            # target_embedding 차원 확인
            if not isinstance(target_embedding, np.ndarray):
                target_embedding = np.array(target_embedding)
                
            # 차원 불일치 확인 및 조정
            if len(valid_embeddings) > 0 and target_embedding.shape[0] != valid_embeddings[0].shape[0]:
                logger.warning(f"타겟 임베딩 차원({target_embedding.shape[0]})이 " 
                            f"제품 임베딩 차원({valid_embeddings[0].shape[0]})과 일치하지 않습니다.")
                
                # 공통 특성 텍스트 재생성 및 임베딩 재계산
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
            
            # 다양성 점수 계산
            main_accord_diversity = np.array([
                0.1 if str(info["mainAccord"]) not in [str(acc) for acc in common_features["main_accords"]] else 0.0
                for info in valid_product_info
            ])
            
            # 스파이스 다양성
            if common_features["spices"]:
                spice_diversity = np.array([
                    self._calculate_spice_diversity(tuple(info["spices"]), tuple(common_features["spices"]))
                    for info in valid_product_info
                ])
            else:
                spice_diversity = np.zeros(len(valid_product_info))
            
            # 최종 점수 계산
            final_scores = (similarities * 0.75) + ((main_accord_diversity + spice_diversity) * 0.25)
            
            # 상위 n개 인덱스 찾기 (유효 데이터 수와 비교)
            top_n = min(top_n, len(valid_product_info))
            if top_n == 0:
                return []
                
            top_indices = np.argsort(final_scores)[-top_n:][::-1]
            
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
            logger.error(f"_find_similar_perfumes_simple 예외: {str(e)}", exc_info=True)
            logger.error(f"common_features: {common_features}")
            raise
    
    @lru_cache(maxsize=128)
    def _calculate_spice_diversity(self, product_spices_tuple, common_spices_tuple):
        """스파이스 다양성 점수 계산 (캐싱 적용)"""
        # 튜플을 리스트로 변환 (필요시)
        product_spices = list(product_spices_tuple) if isinstance(product_spices_tuple, tuple) else product_spices_tuple
        common_spices = list(common_spices_tuple) if isinstance(common_spices_tuple, tuple) else common_spices_tuple
        
        # 교집합 계산
        common = set(str(s) for s in product_spices) & set(str(s) for s in common_spices)
        
        if common_spices:
            spice_overlap_ratio = len(common) / len(common_spices)
            return 0.1 * (1 - spice_overlap_ratio)
        return 0.0