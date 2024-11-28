from typing import Optional, Tuple, List, Dict
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService, prompt_loader: PromptLoader):
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.prompt_loader = prompt_loader

    def process_input(self, user_input: str) -> Tuple[str, Optional[dict]]:
        """
        사용자 입력을 분석하여 대화 또는 추천 모드를 결정합니다.
        """
        if "안녕" in user_input or "대화" in user_input:
            return "chat", None
        
        if "향수" in user_input or any(keyword in user_input for keyword in ["머스크", "시트러스", "우디", "플로럴"]):
            perfume_data = self.db_service.fetch_perfume_data()
            logger.info(f"Fetched perfumes: {perfume_data}")
            if not perfume_data:
                raise ValueError("향수 데이터가 비어 있습니다.")
            return "recommendation", perfume_data

        return "chat", None

    def generate_chat_response(self, user_input: str) -> str:
        """
        JSON 프롬프트를 기반으로 대화 응답을 생성합니다.
        """
        try:
            prompt = self.prompt_loader.get_prompt("chat")
            rules_text = "\n".join(prompt["rules"])
            examples = "\n".join(
                [f"Q: {example['user_input']}\nA: {example['response']}" for example in prompt["examples"]]
            )

            final_prompt = (
                f"{prompt['description']}\nRules:\n{rules_text}\nExamples:\n{examples}\n"
                f"Q: {user_input}\nA:"
            )
            return self.gpt_client.generate_response(final_prompt)
        except Exception as e:
            raise RuntimeError(f"Chat 응답 생성 오류: {e}")

    def generate_recommendation_response(self, user_input: str, perfumes: List[Dict]) -> dict:
        """
        JSON 프롬프트를 기반으로 향수 추천 응답을 생성합니다.
        """
        if not perfumes:
            raise ValueError("향수 데이터가 비어 있습니다.")
        try:
            template = self.prompt_loader.get_prompt("recommendation")
            perfumes_text = "\n".join(
                [f"{perfume['name']}: {perfume['description']}" for perfume in perfumes]
            )
            final_prompt = template["example_prompt"].format(
                user_input=user_input, perfumes_text=perfumes_text
            )
            recommendation = self.gpt_client.generate_response(final_prompt)

            common_feeling_prompt = template["common_feeling_prompt"].format(
                recommendation=recommendation
            )
            common_feeling = self.gpt_client.generate_response(common_feeling_prompt)

            return {
                "recommendation": recommendation,
                "common_feeling": common_feeling
            }
        except Exception as e:
            raise RuntimeError(f"추천 응답 생성 오류: {e}")
        
    