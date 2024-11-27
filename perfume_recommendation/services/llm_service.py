from typing import Optional, Tuple, List, Dict
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader

class LLMService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService, prompt_loader: PromptLoader):
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.prompt_loader = prompt_loader

    def process_input(self, user_input: str) -> Tuple[str, Optional[dict]]:
        if "안녕" in user_input or "대화" in user_input:
            return "chat", None
        elif "향수" in user_input or any(keyword in user_input for keyword in ["머스크", "시트러스", "우디"]):
            return "recommendation", self.db_service.fetch_perfume_data()
        else:
            raise ValueError("적절한 입력을 제공해주세요.")

    def generate_chat_response(self, user_input: str) -> str:
        try:
            prompt = self.prompt_loader.get_prompt("chat")["example_prompt"]
            final_prompt = f"{prompt}\n{user_input}"
            response = self.gpt_client.get_response(final_prompt, "")
            return response
        except Exception as e:
            raise RuntimeError(f"Chat 응답 생성 오류: {e}")

    def generate_recommendation_response(self, user_input: str, perfumes: List[Dict]) -> dict:
        if not perfumes:
            raise ValueError("향수 데이터가 비어 있습니다.")
        try:
            template = self.prompt_loader.get_prompt("recommendation")
            perfumes_text = "\n".join(
                [f"{perfume.get('name', 'Unknown')}: {perfume.get('description', 'No description')}" for perfume in perfumes]
            )
            final_prompt = template["example_prompt"].format(user_input=user_input, perfumes_text=perfumes_text)
            recommendation = self.gpt_client.get_response(final_prompt, perfumes_text)

            common_feeling_prompt = f"""
            추천된 향수들의 공통된 느낌을 요약하세요:
            {recommendation}
            """
            common_feeling = self.gpt_client.get_response(common_feeling_prompt, "")

            return {
                "recommendation": recommendation,
                "common_feeling": common_feeling
            }
        except Exception as e:
            raise RuntimeError(f"추천 응답 생성 오류: {e}")
