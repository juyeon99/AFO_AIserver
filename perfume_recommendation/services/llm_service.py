from typing import Tuple, Optional
import json, logging , re
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
            logger.info(f"Received user input: {user_input}")  # 입력 로그

            # 의도 분류
            intent_prompt = (
                f"사용자의 입력을 영어로 분석하여 의도를 분류하세요:\n"
                f"입력: {user_input}\n"
                f"의도: (1) 향수 추천, (2) 일반 대화"
            )
            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Generated intent prompt: {intent_prompt}")  # 프롬프트 출력
            logger.info(f"Detected intent: {intent}")  # 의도 감지 결과

            if "1" in intent:
                # 향 계열 키워드 추출
                line_extraction_prompt = (
                    f"{user_input} 은 영어로 받아야 합니다. 다음 입력에서 관련성 높은 하나의 향 계열 키워드를 반환하세요. 가능한 키워드는 다음과 같습니다:\n"
                    f"스파이시, 프루티, 시트러스, 그린, 알데히드, 아쿠아틱, 푸제르, 구르망드, 우디, 오리엔탈, 플로럴, 머스크, 파우더리, 앰버, 타바코 레더.\n"
                    f"입력: {user_input}\n결과:"
                )
                raw_line_name = self.gpt_client.generate_response(line_extraction_prompt).strip()
                logger.info(f"Generated line extraction prompt: {line_extraction_prompt}")  # 프롬프트 출력
                logger.info(f"Extracted line name: {raw_line_name}")  # 추출된 계열 이름

                line_name_cleaned = raw_line_name.strip()
                if self.line_mapping.is_valid_line(line_name_cleaned):
                    line_id = self.line_mapping.get_line_id(line_name_cleaned)
                    logger.info(f"Validated line name '{line_name_cleaned}' with line_id: {line_id}")  # 유효성 검사 통과
                    return "recommendation", line_id
                else:
                    logger.error(f"Invalid line name extracted: {raw_line_name}")
                    raise ValueError(f"Invalid line name: {raw_line_name}")

            return "chat", None

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to classify user intent.")

    def generate_recommendation_response(self, user_input: str, line_id: int) -> dict:
        try:
            logger.info(f"Processing recommendation for user input: {user_input}, line_id: {line_id}")

            # Fetch spices for the given line_id
            spices = self.db_service.fetch_spices_by_line(line_id)
            if not spices:
                logger.error(f"No spices found for line_id {line_id}.")
                raise ValueError("No spices found for the given fragrance line.")

            logger.info(f"Fetched spices for line_id {line_id}: {spices}")

            # Fetch perfumes using the spices
            perfumes = self.db_service.fetch_perfumes_by_spices(spices)
            if not perfumes:
                logger.error(f"No perfumes found for spices: {spices}")
                raise ValueError("No perfumes found for the given spices.")

            logger.info(f"Fetched perfumes for spices {spices}: {perfumes}")

            # Prepare perfumes text for GPT prompt
            perfumes_text = "\n".join([
                f"{idx + 1}. {perfume['perfume_id']} ({perfume['perfume_brand']}): {perfume['perfume_description']}"
                for idx, perfume in enumerate(perfumes)
            ])
            logger.info(f"Prepared perfumes text for GPT prompt:\n{perfumes_text}")

            # Load prompt template and generate final GPT prompt
            template = self.prompt_loader.get_prompt("recommendation")
            final_prompt = (
                f"{template['description']}\n"
                f"Perfumes:\n{perfumes_text}\n"
                f"Return the result as a JSON object with a 'recommendations' key. "
                f"The 'recommendations' key must be a list of objects, each containing 'reason' and 'situation'."
            )
            logger.info(f"Generated GPT prompt:\n{final_prompt}")

            # Generate response from GPT
            response_text = self.gpt_client.generate_response(final_prompt).strip()
            if not response_text:
                logger.error("GPT response is empty.")
                raise ValueError("Received an empty response from GPT.")

            # Extract JSON structure from GPT response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                logger.error(f"Invalid JSON structure in GPT response: {response_text}")
                raise ValueError("Invalid JSON structure in GPT response.")

            gpt_response = json.loads(response_text[json_start:json_end])

            # Validate 'recommendations' key
            if "recommendations" not in gpt_response:
                logger.warning(f"'recommendations' key missing in GPT response: {gpt_response}")
                gpt_response["recommendations"] = [
                    {"reason": "Default reason: Unable to generate detailed recommendations.",
                    "situation": "Default situation: Suitable for general occasions."}
                ]

            # Generate recommendations
            recommendations = [
                {
                    "name": f"{perfume['perfume_id']} ({perfume['perfume_brand']})",
                    "reason": gpt_entry.get("reason", "No specific reason provided"),
                    "situation": gpt_entry.get("situation", "No specific situation provided")
                }
                for perfume, gpt_entry in zip(perfumes, gpt_response["recommendations"])
            ]
            logger.info(f"Final recommendations: {recommendations}")

            # Generate common feeling
            common_feeling_prompt = template["common_feeling_prompt"].format(recommendation=recommendations)
            common_feeling = self.gpt_client.generate_response(common_feeling_prompt).strip()

            # Generate image prompt
            image_generation_prompt = f"Translate the following text into English for an image generation prompt: {common_feeling}"
            image_prompt = self.gpt_client.generate_response(image_generation_prompt).strip()

            # Return structured response
            return {
                "recommendations": recommendations,
                "common_feeling": common_feeling,
                "image_prompt": image_prompt,
            }
        except ValueError as e:
            logger.error(f"ValueError in recommendation generation: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unhandled error in generate_recommendation_response: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations.")

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
        try:
            logger.info(f"Processing recommendation for user input: {user_input}, line_id: {line_id}")

            # Validate inputs
            if not user_input or not line_id:
                raise ValueError("User input or line_id is missing or invalid.")

            # Fetch spices for the given line_id
            spices = self.db_service.fetch_spices_by_line(line_id)
            if not spices:
                logger.error(f"No spices found for line_id {line_id}.")
                raise ValueError("No spices found for the given fragrance line.")

            logger.info(f"Fetched spices for line_id {line_id}: {spices}")

            # Fetch perfumes using the spices
            perfumes = self.db_service.fetch_perfumes_by_spices(spices)
            if not perfumes:
                logger.error(f"No perfumes found for spices: {spices}")
                raise ValueError("No perfumes found for the given spices.")

            logger.info(f"Fetched perfumes for spices {spices}: {perfumes}")

            # Prepare perfumes text for GPT prompt
            perfumes_text = "\n".join([  
                f"{idx + 1}. {perfume['perfume_id']} ({perfume['perfume_brand']}): {perfume['perfume_description']}"  
                for idx, perfume in enumerate(perfumes)  
            ])  
            logger.info(f"Prepared perfumes text for GPT prompt:\n{perfumes_text}")

            # Load prompt template and generate final GPT prompt
            template = self.prompt_loader.get_prompt("recommendation")
            final_prompt = (  
                f"{template['description']}\n"  
                f"Perfumes:\n{perfumes_text}\n"  
                f"Return the result as a JSON object with a 'recommendations' key. "  
                f"The 'recommendations' key must be a list of objects, each containing 'reason' and 'situation'."  
            )  
            logger.info(f"Generated GPT prompt:\n{final_prompt}")

            # Generate response from GPT for perfume recommendations
            response_text = self.gpt_client.generate_response(final_prompt).strip()
            if not response_text:
                logger.error("GPT response is empty.")
                raise ValueError("Received an empty response from GPT.")

            # Extract JSON structure from GPT response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                logger.error(f"Invalid JSON structure in GPT response: {response_text}")
                raise ValueError("Invalid JSON structure in GPT response.")

            gpt_response = json.loads(response_text[json_start:json_end])

            # Validate 'recommendations' key
            if "recommendations" not in gpt_response:
                logger.warning(f"'recommendations' key missing in GPT response: {gpt_response}")
                gpt_response["recommendations"] = [
                    {"reason": "Default reason: Unable to generate detailed recommendations.",
                    "situation": "Default situation: Suitable for general occasions."}
                ]

            # Generate recommendations
            recommendations = [
                {
                    "id": f"{perfume['perfume_id']}",
                    "reason": gpt_entry.get("reason", "No specific reason provided"),
                    "situation": gpt_entry.get("situation", "No specific situation provided")
                }
                for perfume, gpt_entry in zip(perfumes, gpt_response["recommendations"])
            ]
            logger.info(f"Final recommendations: {recommendations}")

            # Generate common feeling
            common_feeling_prompt = template["common_feeling_prompt"].format(recommendation=recommendations)

            # 정규 표현식을 사용하여 숫자 또는 번호를 제거 (숫자만 추출해서 제거)
            common_feeling_prompt = re.sub(r'\d+', '', common_feeling_prompt)

            # GPT 클라이언트를 사용하여 응답 생성
            common_feeling = self.gpt_client.generate_response(common_feeling_prompt).strip()

            # GPT에게 이미지 생성 요청을 위한 프롬프트 생성
            image_prompt = f"Create a detailed description of an image based on the following keywords: {user_input}"

            # GPT에게 이미지 생성을 위한 텍스트 요청
            image_description = self.gpt_client.generate_response(image_prompt).strip()

            # Return structured response with image description from GPT
            return {
                "recommendations": recommendations,
                "common_feeling": common_feeling,
                "image_prompt": image_description,
                "line_id": line_id
            }
        except ValueError as e:
            logger.error(f"ValueError in recommendation generation: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unhandled error in generate_recommendation_response: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations.")