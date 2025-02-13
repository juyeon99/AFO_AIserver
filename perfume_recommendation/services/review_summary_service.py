from sqlalchemy.orm import Session
from perfume_recommendation.models.base_model import Review
from models.client import GPTClient
from cachetools import TTLCache # 캐시 만료 시간 설정을 위한 라이브러리
from datetime import datetime
from typing import Tuple

# 24시간 TTL을 가진 캐시 생성
summary_cache = TTLCache(maxsize=100, ttl=24 * 3600)

class ReviewService:
    def __init__(self):
        self.gpt_client = GPTClient()
        # 갱신 기준 설정
        self.UPDATE_THRESHOLD = {
            'count': 5,  # 새로운 리뷰 수 기준
            'percentage': 0.1  # 전체 대비 새로운 리뷰 비율 (10%)
        }
    
    async def get_review_summary(self, product_id: int, db: Session) -> str:
        """
        제품의 리뷰 요약을 가져오는 메서드
        
        Args:
            product_id (int): 제품 ID
            db (Session): 데이터베이스 세션
            
        Returns:
            str: 리뷰 요약 텍스트
        """
        # 가장 최근 리뷰 확인
        latest_review = db.query(Review)\
            .filter(Review.product_id == product_id)\
            .order_by(Review.time_stamp.desc())\
            .first()
            
        if not latest_review:
            return "리뷰가 없습니다."

        # 버전 정보가 포함된 캐시 키
        cache_key = f"summary_v1_{product_id}"
        
        # 캐시된 데이터 확인
        cached_data = summary_cache.get(cache_key)
        if cached_data:
            should_update = await self._check_update_needed(
                product_id, 
                db, 
                cached_data
            )
            
            if not should_update:
                cached_summary, _, _ = cached_data
                return cached_summary

        # 모든 리뷰 가져오기
        reviews = db.query(Review)\
            .filter(Review.product_id == product_id)\
            .all()
            
        review_texts = [review.content for review in reviews]
        
        # GPT 프롬프트 구성
        summary = await self._generate_summary(review_texts)
        
        # 캐시 업데이트
        summary_cache[cache_key] = (
            summary, 
            latest_review.time_stamp,
            len(review_texts)
        )
        
        return summary

    async def _check_update_needed(
        self, 
        product_id: int, 
        db: Session, 
        cached_data: Tuple[str, datetime, int]
        ) -> bool:
        """
        캐시 업데이트가 필요한지 확인하는 메서드
        
        Args:
            product_id (str): 제품 ID
            db (Session): 데이터베이스 세션
            cached_data (Tuple[str, datetime, int]): 캐시된 데이터 (요약, 타임스탬프, 리뷰 수)
            
        Returns:
            bool: 업데이트 필요 여부
        """
        # 캐시된 데이터에서 필요한 정보 추출
        _, cached_timestamp, cached_review_count = cached_data
        
        # 현재 전체 리뷰 수 확인
        current_total_reviews = db.query(Review)\
            .filter(Review.product_id == product_id)\
            .count()
            
        # 리뷰가 삭제되었는지 확인 (현재 리뷰 수가 캐시된 리뷰 수보다 적으면 삭제된 것)
        if current_total_reviews < cached_review_count:
            return True
        
        # cached_timestamp를 datetime 객체로 변환 (문자열인 경우)
        if isinstance(cached_timestamp, str):
            try:
                cached_timestamp = datetime.strptime(cached_timestamp, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                # 날짜 형식이 맞지 않으면 업데이트 필요
                return True
        
        # 새로운 리뷰 수 확인 (캐시된 타임스탬프 이후의 리뷰들)
        new_reviews_count = db.query(Review)\
            .filter(Review.product_id == product_id)\
            .filter(Review.time_stamp > cached_timestamp)\
            .count()
                
        # 갱신 조건 확인
        should_update = (
            new_reviews_count >= self.UPDATE_THRESHOLD['count'] or  # 새 리뷰가 5개 이상
            (current_total_reviews > 0 and 
            new_reviews_count / current_total_reviews >= self.UPDATE_THRESHOLD['percentage'])  # 새 리뷰가 전체의 10% 이상
        )
        
        return should_update

    async def _generate_summary(self, review_texts: list[str]) -> str:
        """
        리뷰 요약을 생성하는 메서드
        
        Args:
            review_texts (list[str]): 리뷰 텍스트 리스트
            
        Returns:
            str: 생성된 요약 텍스트
        """
        prompt = f"""총 {len(review_texts)}개의 리뷰를 분석하여 핵심만 간단히 요약해주세요.
        다음 내용을 2-3줄로 간단히 작성해주세요:

        - 대다수 사용자들의 주요 평가
        - 근거가 없는 내용은 제외
        - 일부 사용자들의 참고할만한 의견
        - 어떤 사용자에게 적합한지

        리뷰들:
        {chr(10).join(f"- {text}" for text in review_texts)}"""

        return await self.gpt_client.generate_response(prompt)