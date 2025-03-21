import json
import logging
from typing import Dict, List, Tuple
from models.client import GPTClient
from services.db_service import DBService
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Define therapy titles for each category
THERAPY_HASHTAGS = {
    0: {"korean": "#숙면_유도 #깊은_휴식", "english": "#inducing_deep_sleep #deep_rest"},
    1: {"korean": "#업무_효율 #생산성_증진", "english": "#work_efficiency #productivity_boost"},
    2: {"korean": "#상쾌한_아침 #활기찬_하루", "english": "#refreshing_mornings #energetic_day"},
    3: {"korean": "#마음의_평화 #내적_안정", "english": "#inner_peace #mental_stability"},
    4: {"korean": "#감정_균형 #안정적인_마음", "english": "#emotional_balance #stable_mind"},
    5: {"korean": "#공기_청정 #깨끗한_환경", "english": "#air_purification #clean_environment"}
}

class DiffuserRecommendationService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService) -> None:
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.DIFFUSER_CATEGORY_ID = 2
        
        # 카테고리별 기본 향료 정보 (GPT 프롬프트용)
        # self.user_input_info = {
        #     "수면 & 회복": "수면과 휴식에 도움을 주는 진정 효과가 있는 향료들입니다. 라벤더, 캐모마일 등이 대표적입니다.",
        #     "집중 & 마인드풀니스": "집중력 향상과 맑은 정신에 도움을 주는 향료들입니다. 로즈마리, 페퍼민트 등이 효과적입니다.",
        #     "활력 & 에너지": "활력과 에너지를 북돋아주는 상쾌한 향료들입니다. 시트러스 계열이 대표적입니다.",
        #     "평온 & 스트레스 해소": "스트레스 해소와 마음의 안정에 도움을 주는 향료들입니다. 라벤더, 일랑일랑 등이 효과적입니다.",
        #     "기쁨 & 긍정": "긍정적인 기분과 행복감을 고취시키는 향료들입니다. 오렌지, 바닐라 등이 대표적입니다.",
        #     "리프레시 & 클린 에어": "공간을 상쾌하고 깨끗하게 만들어주는 향료들입니다. 유칼립투스, 레몬 등이 효과적입니다."
        # }
        self.category = {
            0: "Sleep & Recovery",
            1: "Focus & Mindfulness",
            2: "Vitality & Energy",
            3: "Calm & Stress Relief",
            4: "Joy & Positivity",
            5: "Refresh & Clean Air"
        }

        self.description = {
            0: "Ingredients that promote sleep and relaxation, such as lavender, chamomile, etc.",
            1: "Ingredients that help improve concentration and mental clarity, such as rosemary, peppermint, etc.",
            2: "Refreshing ingredients that boost vitality and energy, such as spices in the citrus family.",
            3: "Ingredients that help relieve stress and calm the mind, such as lavender, ylang-ylang, etc.",
            4: "Ingredients that promote positive emotions and happiness, such as orange, vanilla, etc.",
            5: "Ingredients that help refresh and clean the air, such as eucalyptus, lemon, etc."
        }

    async def get_recommended_notes(self, category_index: int) -> List[str]:
        """GPT를 통해 유저 입력에 맞는 최적의 향료 조합 추천"""
        prompt = f"""
        당신은 아로마테라피와 디퓨저 전문가입니다. 주어진 목적에 가장 적합한 향료 조합을 추천해주세요.

        목적: {self.category[category_index]}
        설명: {self.description[category_index]}

        다음 사항들을 고려하여 향료 조합을 추천해주세요:
        1. 주요 효과: 해당 목적을 달성하는데 가장 효과적인 향료들
        2. 향료 조화: 서로 잘 어울리는 향료 조합
        3. 강도 균형: 향의 강도가 적절히 조화를 이루는 조합
        4. 지속 효과: 효과가 잘 지속될 수 있는 조합

        아래 향료들 중에서 4-6개를 선택하여 최적의 조합을 만들어주세요:
        베티버, 클라리 세이지, 라벤더, 일랑일랑, 바닐라, 캐모마일, 로즈마리, 레몬, 
        페퍼민트, 유칼립투스, 베르가못, 자몽, 레몬그라스, 오렌지, 제라늄, 텐저린, 
        자스민, 사이프러스, 파인, 티트리, 라반딘

        JSON 형식으로 응답:
        {{"selected_notes": ["향료1", "향료2", "향료3", "향료4"]}}
        """
        
        try:
            response = await self.gpt_client.generate_response(prompt)
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            result = json.loads(response.strip())
            return result["selected_notes"]
        except Exception as e:
            logger.error(f"향료 추천 생성 실패: {str(e)}")
            raise

    async def get_usage_routine(self, category_index: int) -> str:
        """GPT를 통해 사용 루틴 생성"""
        prompt = f"""
        당신은 디퓨저 전문가입니다. 다음 상황에 가장 적합한 구체적인 사용 루틴을 제안해주세요.

        카테고리: {self.category[category_index]}
        설명: {self.description[category_index]}

        다음을 포함하여 구체적인 사용 루틴을 작성해주세요:
        1. 사용 시점 (언제)
        2. 사용 장소 (어디서)
        3. 기대 효과 (왜)

        JSON 형식으로 응답:
        {{"usage_routine": 80자 이내로 향료이름은 제외하고 구체적인 사용 루틴을 작성}}
        """

        response = await self.gpt_client.generate_response(prompt)
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        result = json.loads(response.strip())
        return result["usage_routine"]

    async def recommend_diffusers(self, language: str, category_index: int) -> Dict:
        """카테고리에 맞는 디퓨저 추천"""
        try:
            logger.info(f"language: {language}")
            logger.info(f"Generating recommendation for: {category_index}")
            
            if category_index not in (0, 1, 2, 3, 4, 5):
                raise ValueError("Invalid category")
            
            # 1. GPT를 통해 향료 조합 추천 받기
            recommended_notes = await self.get_recommended_notes(category_index)
            
            # 2. 추천받은 향료들로 디퓨저 검색
            spices = self.db_service.get_spices_by_names(recommended_notes)
            if not spices:
                raise ValueError("추천할 수 있는 향료가 없습니다")

            # 3. 해당 향료들이 포함된 디퓨저 찾기
            spice_ids = [spice['id'] for spice in spices]
            diffusers = self.db_service.get_diffusers_by_spice_ids(spice_ids)
            
            if not diffusers:
                raise ValueError("추천할 수 있는 디퓨저가 없습니다")
            
            # 4. 사용 루틴 생성
            usage_routine = await self.get_usage_routine(category_index)

            # 5. 최종 응답 구성
            recommendations = [
                {
                    'product_id': diffuser['id'],
                    'name': f"{diffuser['name_kr']} {diffuser.get('volume', '200ml')}",
                    'brand': diffuser['brand'],
                    'content': diffuser['content']
                }
                for diffuser in diffusers[:2]
            ]

            return {
                'recommendations': recommendations,
                'usage_routine': usage_routine,
                'therapy_title': THERAPY_HASHTAGS[category_index][language]
            }

        except Exception as e:
            logger.error(f"추천 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"추천 생성에 실패했습니다: {str(e)}"
            )