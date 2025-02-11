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
        self.DIFFUSER_CATEGORY_ID = 2
        
        # 카테고리별 추천 향료 미리 정의
        self.user_input_notes = {
            "수면 & 회복": ["베티버","클라리 세이지", "라벤더", "일랑일랑", "바닐라", "캐모마일"],
            "집중 & 마인드풀니스": ["로즈마리", "레몬", "페퍼민트", "유칼립투스"],
            "활력 & 에너지": ["베르가못", "자몽", "레몬그라스", "오렌지"],
            "평온 & 스트레스 해소": ["라벤더", "베르가못", "제라늄", "일랑일랑"],
            "기쁨 & 긍정": ["텐저린", "오렌지", "바닐라", "자스민", "베르가못"],
            "리프레시 & 클린 에어": ["사이프러스", "레몬", "유칼립투스", "파인", "티트리","라반딘"]
        }

    async def recommend_diffusers(self, user_input: str) -> Dict:
        """카테고리에 맞는 디퓨저 추천"""
        try:
            logger.info(f"Generating recommendation for: {user_input}")
            
            if user_input not in self.user_input_notes:
                raise ValueError("유효하지 않은 카테고리입니다")
            
            # 1. 해당 카테고리의 향료로 디퓨저 검색
            recommended_notes = self.user_input_notes[user_input]
            spices = self.db_service.get_spices_by_names(recommended_notes)
            
            if not spices:
                raise ValueError("추천할 수 있는 향료가 없습니다")

            # 2. 해당 향료들을 포함하는 디퓨저 찾기
            spice_ids = [spice['id'] for spice in spices]
            diffusers = self.db_service.get_diffusers_by_spice_ids(spice_ids)
            
            if not diffusers:
                raise ValueError("추천할 수 있는 디퓨저가 없습니다")
            
            # 3. GPT로 사용 루틴 생성
            prompt = f"""
            당신은 디퓨저 전문가입니다. 다음 상황에 가장 적합한 간단한 사용 루틴을 제안해주세요.

            상황: {user_input}

            아래와 같은 간단한 사용 루틴 형식으로 작성해주세요:
            - 수면 & 회복: "조용한 시간에 디퓨저를 켜고 명상이나 깊은 호흡과 함께 내면의 평화를 찾아보세요."
            - 집중 & 마인드풀니스: "학습이나 업무 시작 전에 디퓨저를 켜고 맑은 정신으로 일에 집중해보세요."
            - 활력 & 에너지: "아침에 기상하여 디퓨저를 켜고 활기찬 하루를 시작해보세요."
            - 평온 & 스트레스 해소: "스트레스가 쌓인 순간, 디퓨저를 켜고 깊은 호흡과 함께 마음의 안정을 찾아보세요."
            - 기쁨 & 긍정: "디퓨저의 밝은 향기와 함께 하루의 긍정적인 순간들을 떠올려보세요."
            - 리프레시 & 클린 에어: "환기 후 디퓨저를 켜서 상쾌하고 깨끗한 공기로 공간을 채워보세요."

            다음 형식의 JSON으로만 응답해주세요:
            {{
                "usage_routine": "100자 이내로 사용 루틴만 작성해주세요. 향료 설명은 제외하고 언제, 어떻게 사용하면 좋은지 구체적으로 설명해주세요."
            }}"""

            # GPT 응답 받기
            response = await self.gpt_client.generate_response(prompt)
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            gpt_result = json.loads(response.strip())

            # 4. 최종 응답 구성
            recommendations = [
                {
                    'product_id': diffuser['id'],
                    'name': f"{diffuser['name_kr']} {diffuser.get('volume', '200ml')}",
                    'brand': diffuser['brand']
                }
                for diffuser in diffusers[:2]
            ]

            return {
                'recommendations': recommendations,
                'usage_routine': gpt_result['usage_routine']
            }

        except Exception as e:
            logger.error(f"추천 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"추천 생성에 실패했습니다: {str(e)}"
            )