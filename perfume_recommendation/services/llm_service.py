from typing import List, Dict, Tuple, Optional
import json , logging
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from utils.line_mapping import LineMapping
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService, prompt_loader: PromptLoader, line_mapping: LineMapping):
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.prompt_loader = prompt_loader
        self.line_mapping = line_mapping

    def process_input(self, user_input: str) -> Tuple[str, Optional[int]]:
        """
        사용자 입력을 분석하여 의도를 분류하고, 향 계열 이름을 추출합니다.
        """
        try:
            # 의도 분류
            intent_prompt = (
                f"사용자의 입력을 분석하여 의도를 분류하세요:\n"
                f"입력: {user_input}\n"
                f"의도: (1) 향수 추천, (2) 일반 대화"
            )
            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Detected intent: {intent}")

            if "1" in intent:
                # 향 계열 키워드 추출 (하나의 계열만 선택)
                line_extraction_prompt = (
                    f"다음 입력에서 관련성 높은 하나의 향 계열 키워드를 반환하세요. 가능한 키워드는 다음과 같습니다:\n"
                    f"스파이시, 프루티, 시트러스, 그린, 알데히드, 아쿠아틱, 푸제르, 구르망드, 우디, 오리엔탈, 플로럴, 머스크, 파우더리, 앰버, 타바코 레더.\n"
                    f"입력: {user_input}\n결과:"
                )
                raw_line_name = self.gpt_client.generate_response(line_extraction_prompt).strip()
                logger.info(f"Extracted line name: {raw_line_name}")

                # 유효한 계열 이름에서 ID 추출
                line_name_cleaned = raw_line_name.strip()  # 공백 제거
                if self.line_mapping.is_valid_line(line_name_cleaned):
                    line_id = self.line_mapping.get_line_id(line_name_cleaned)
                    logger.info(f"Using line_id: {line_id}")
                    return "recommendation", line_id
                else:
                    # Extracted line name was invalid
                    logger.error(f"Invalid line name extracted: {raw_line_name}")
                    raise ValueError(f"Invalid line name: {raw_line_name}")

            # If the intent was not a recommendation, return "chat" mode
            return "chat", None

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to classify user intent.")

    def generate_chat_response(self, user_input: str) -> str:
        """
        JSON 프롬프트를 기반으로 대화 응답을 생성합니다.
        """
        try:
            prompt = self.prompt_loader.get_prompt("chat")
            rules_text = "\n".join(prompt.get("rules", []))
            examples = "\n".join([
                f"Q: {example.get('user_input', 'No input')}\nA: {example.get('response', 'No response')}"
                for example in prompt.get("examples", [])
            ])
            final_prompt = (
                f"{prompt.get('description', '')}\nRules:\n{rules_text}\nExamples:\n{examples}\n"
                f"Q: {user_input}\nA:"
            )
            response = self.gpt_client.generate_response(final_prompt).strip()
            logger.info(f"Generated chat response: {response}")
            return response
        except Exception as e:
            logger.error(f"Chat response generation error for input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to generate chat response.")

    def generate_recommendation_response(self, user_input: str, line_id: int) -> dict:
            """
            향수를 추천하고 관련 정보를 생성합니다.
            """
            try:
                # line_id로 해당 계열의 spice 목록을 가져옵니다.
                spices = self.db_service.fetch_spices_by_line(line_id)
                if not spices:
                    logger.error(f"No spices found for line_id: {line_id}")
                    raise ValueError(f"No spices available for line_id: {line_id}")

                logger.info(f"Processing recommendation for spices: {spices}")

                # 향수 데이터를 가져옴
                perfumes = self.db_service.fetch_perfumes_by_spices(spices)
                if not perfumes:
                    logger.error(f"No perfumes found for the spices: {spices}")
                    raise ValueError(f"No perfumes available for the requested spices: {spices}")

                # 향수 리스트를 텍스트로 변환
                perfumes_text = "\n".join([
                    f"- {perfume['perfume_name']} ({perfume['perfume_brand']}): {perfume['perfume_description']} "
                    f"- 주요 향료: {perfume['spice_name']} - 이미지 URL: {perfume['perfume_url']}"
                    for perfume in perfumes
                ])
                template = self.prompt_loader.get_prompt("recommendation")
                rules_text = '\n'.join(template["rules"])
                json_example = json.dumps(template["examples"][0]["response"], ensure_ascii=False, indent=2)

                # 프롬프트 생성
                final_prompt = (
                    f"{template['description']}\n"
                    f"Rules:\n{rules_text}\n"
                    f"향수 목록:\n{perfumes_text}\n"
                    f"사용자 요청: {user_input}\n"
                    f"응답은 항상 아래 JSON 형식을 따르세요:\n{json_example}"
                )

                response_text = self.gpt_client.generate_response(final_prompt).strip()
                recommendation = json.loads(response_text)

                # 공통 느낌 생성
                common_feeling_prompt = template["common_feeling_prompt"].format(recommendation=recommendation)
                common_feeling = self.gpt_client.generate_response(common_feeling_prompt).strip()

                # 이미지 생성 프롬프트를 영어로 변환
                image_generation_prompt = f"Translate the following text into English for an image generation prompt: {common_feeling}"
                translated_prompt = self.gpt_client.generate_response(image_generation_prompt).strip()

                logger.info(f"Generated recommendation: {recommendation}")
                return {
                    "recommendation": recommendation,
                    "common_feeling": common_feeling,
                    "image_prompt": translated_prompt  # 번역된 영어 프롬프트 사용
                }
            except ValueError as ve:
                logger.error(f"Recommendation error: {ve}")
                raise HTTPException(status_code=400, detail=f"Recommendation error: {ve}")
            except Exception as e:
                logger.error(f"Unhandled recommendation generation error: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate recommendation.")