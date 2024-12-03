from typing import Tuple, Optional
import json, logging
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
                    logger.error(f"Invalid line name extracted: {raw_line_name}")
                    raise ValueError(f"Invalid line name: {raw_line_name}")

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
            # 데이터베이스에서 향료와 향수 정보 가져오기
            spices = self.db_service.fetch_spices_by_line(line_id)
            if not spices:
                logger.error(f"No spices found for line_id: {line_id}")
                raise ValueError(f"No spices available for line_id: {line_id}")

            perfumes = self.db_service.fetch_perfumes_by_spices(spices)
            if not perfumes:
                logger.error(f"No perfumes found for the spices: {spices}")
                raise ValueError(f"No perfumes available for the requested spices: {spices}")

            # 향수 정보 텍스트 생성
            perfumes_text = "\n".join([
                f"{perfume['perfume_name']}: {perfume['perfume_description']}"
                for perfume in perfumes
            ])

            # 프롬프트 템플릿 가져오기
            template = self.prompt_loader.get_prompt("recommendation")
            json_example = json.dumps(template["examples"][0]["response"], ensure_ascii=False, indent=2)

            # 최종 프롬프트 생성
            final_prompt = (
                f"{template['description']}\n"
                f"Rules:\n" + "\n".join(template["rules"]) + "\n"
                f"향수 목록:\n{perfumes_text}\n"
                f"사용자 요청: {user_input}\n"
                f"응답은 반드시 아래와 같은 완벽한 JSON 형식이어야 합니다:\n{json_example}"
            )

            # GPT 응답 생성 및 로깅
            logger.info("Sending prompt to GPT...")
            response_text = self.gpt_client.generate_response(final_prompt)
            logger.info(f"Raw GPT response (length: {len(response_text)}): {response_text}")

            if not response_text or not response_text.strip():
                logger.error("GPT returned empty response")
                raise ValueError("GPT 응답이 비어있습니다")

            # JSON 파싱
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    logger.error(f"No valid JSON found in response: {response_text}")
                    raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")
                    
                json_text = response_text[json_start:json_end]
                logger.info(f"Extracted JSON text: {json_text}")
                
                recommendation = json.loads(json_text)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error. Response: {response_text}")
                logger.error(f"Parse error details: {str(e)}")
                raise ValueError(f"GPT 응답을 JSON으로 파싱할 수 없습니다: {str(e)}")

            # 공통 느낌 생성
            common_feeling_prompt = f"Describe the overall feeling of the recommended perfumes in English with a focus on nature or landscapes. {recommendation}"
            common_feeling = self.gpt_client.generate_response(common_feeling_prompt)

            # 이미지 프롬프트 생성
            image_generation_prompt = f"Generate an image based on the following natural or landscape feeling: {common_feeling}"

            return {
                "recommendation": recommendation,
                "common_feeling": common_feeling,
                "image_prompt": image_generation_prompt
            }

        except ValueError as ve:
            logger.error(f"Recommendation error: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Unhandled recommendation generation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendation.")

    def generate_image_prompt(self, user_input: str) -> str:
        """
        사용자 입력을 분석하여 감정이나 상황을 표현하는 이미지 프롬프트를 생성합니다.
        짧은 문장을 받아도 반드시 이미지 생성을 할 수 있도록 문장을 확장합니다.
        """
        try:
            emotion_prompt = (
                f"사용자의 입력을 분석하여 감정이나 분위기를 파악하세요.\n"
                f"입력: {user_input}\n"
                f"결과: 감정 또는 분위기를 간단하게 묘사하십시오 (예: 행복, 차분함, 슬픔 등). "
                f"이 감정에 맞는 이미지의 구체적인 디테일을 설명하십시오. "
                f"예를 들어, '행복'이면 밝고 따뜻한 색조와 자연 풍경, 사람들의 웃음과 같은 특징을 포함하도록."
            )
            emotion = self.gpt_client.generate_response(emotion_prompt).strip()
            logger.info(f"Detected emotion: {emotion}")

            # 감정과 분위기에 맞는 이미지를 구체적으로 설명하도록 확장
            image_prompt = (
                f"Generate an image based on the following emotion or atmosphere: {emotion}. "
                f"Include details such as lighting, color tones, the setting, objects, and people, "
                f"ensuring the scene visually captures the feeling described in the emotion."
            )
            
            return image_prompt

        except Exception as e:
            logger.error(f"Error generating image prompt for input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to generate image prompt.")
