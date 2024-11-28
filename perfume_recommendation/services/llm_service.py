from typing import Optional, Tuple, List, Dict
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader
import logging
import json

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService, prompt_loader: PromptLoader):
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.prompt_loader = prompt_loader

    def process_input(self, user_input: str) -> Tuple[str, Optional[dict]]:
        """
        LLM을 활용해 의도를 분류하고, 추천 모드인지 대화 모드인지 결정합니다.
        """
        try:
        # 의도 분류를 위한 간단한 프롬프트 생성
            intent_prompt = (
            f"사용자의 입력을 분석하여 의도를 분류하세요:\n"
            f"입력: {user_input}\n"
            f"의도: (1) 향수 추천, (2) 일반 대화"
            )
            # LLM 호출
            intent = self.gpt_client.generate_response(intent_prompt)
            logger.info(f"Detected intent: {intent}")

            # 추천 의도로 판별된 경우
            if "1" in intent:
                perfume_data = self.db_service.fetch_perfume_data()
                logger.info(f"Fetched perfumes: {perfume_data}")
                if not perfume_data:
                    raise ValueError("향수 데이터가 비어 있습니다.")
                return "recommendation", perfume_data

            # 기본값은 대화 모드
            return "chat", None
        except Exception as e:
            logger.error(f"의도 분류 오류: {e}")
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
            
            json_example = json.dumps(template["examples"][0]["response"], ensure_ascii=False, indent=2)
            logger.info(f"Generated JSON Example: {json_example}")
            
            final_prompt = (
                template["description"] + "\n" +
                "Rules:\n" + "\n".join(template["rules"]) + "\n" +
                "향수 목록:\n" + perfumes_text + "\n" +
                "사용자 요청: " + user_input + "\n" +
                "응답은 항상 아래 JSON 형식을 따르세요:\n" +
                json_example
            )
            final_prompt += json_example 
            
            logger.info(f"Generated Final Prompt: {final_prompt}")
            # GPT로 응답 생성
            response_text = self.gpt_client.generate_response(final_prompt)

            # 모델의 응답을 JSON으로 파싱
            recommendation = json.loads(response_text)

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
        
    