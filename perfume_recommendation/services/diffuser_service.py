import json
import logging
from typing import Dict
from models.client import GPTClient
from services.db_service import DBService
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class DiffuserRecommendationService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService) -> None:
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.DIFFUSER_CATEGORY_ID = 2  # 디퓨저 카테고리 ID

    async def get_diffuser_data(self):
        """디퓨저 데이터만 가져오기"""
        try:
            # category_id로 디퓨저만 필터링하여 가져오기
            all_products = self.db_service.load_cached_product_data()
            diffusers = [
                product for product in all_products 
                if product.get('category_id') == self.DIFFUSER_CATEGORY_ID
            ]
            return diffusers
        except Exception as e:
            logger.error(f"디퓨저 데이터 로드 실패: {str(e)}")
            raise ValueError("디퓨저 데이터를 가져올 수 없습니다")

    async def recommend_diffusers(self, category: str) -> Dict:
        """카테고리에 맞는 디퓨저 추천"""
        try:
            logger.info(f"Generating recommendation for: {category}")
            
            # 1. 디퓨저 데이터만 로드
            diffusers = await self.get_diffuser_data()
            if not diffusers:
                raise ValueError("추천 가능한 디퓨저를 찾을 수 없습니다")

            # 2. GPT 프롬프트 생성
            diffuser_info = "\n".join([
                f"{d['id']}. {d['name']} ({d['brand']}) - 노트: {d['notes']}"
                for d in diffusers[:20]
            ])

            prompt = f"""
당신은 디퓨저 전문가입니다. 사용자의 상황에 가장 적합한 디퓨저 2개를 추천해주세요.

상황: {category}

사용 가능한 디퓨저 목록:
{diffuser_info}

다음 형식의 JSON으로만 응답해주세요:
{{
    "recommendations": [
        {{
            "id": "디퓨저 ID",
            "name": "디퓨저 이름",
            "brand": "브랜드명"
        }},
        {{
            "id": "디퓨저 ID",
            "name": "디퓨저 이름",
            "brand": "브랜드명"
        }}
    ],
    "usage_routine": "이 상황에 맞는 디퓨저 사용 루틴 제안 (50자 이내)"
}}"""

            # 3. GPT 응답 생성
            response = self.gpt_client.generate_response(prompt)
            
            try:
                # JSON 파싱
                if '```json' in response:
                    response = response.split('```json')[1].split('```')[0]
                gpt_result = json.loads(response.strip())

                # 4. 디퓨저 정보 매핑
                recommendations = []
                for rec in gpt_result['recommendations']:
                    diffuser = next(
                        (d for d in diffusers if str(d['id']) == str(rec['id'])),
                        None
                    )
                    if diffuser:
                        recommendations.append({
                            'name': f"{diffuser['name']} {diffuser.get('volume', '200ml')}",
                            'brand': diffuser['brand']
                        })

                # 5. 최종 응답 반환
                return {
                    'recommendations': recommendations,
                    'usage_routine': gpt_result['usage_routine']
                }

            except json.JSONDecodeError as e:
                logger.error(f"GPT 응답 파싱 실패: {e}")
                raise ValueError("추천 생성 중 오류가 발생했습니다")

        except Exception as e:
            logger.error(f"추천 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"추천 생성에 실패했습니다: {str(e)}"
            )