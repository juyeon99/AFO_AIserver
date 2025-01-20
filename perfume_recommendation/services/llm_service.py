import json
import logging
from typing import Tuple, Optional
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService, prompt_loader: PromptLoader):
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.prompt_loader = prompt_loader

    def generate_chat_response(self, user_input: str) -> str:
        """
        Generate a chat response using GPT for general conversation.
        """
        try:
            # Load a chat prompt template
            template = self.prompt_loader.get_prompt("chat")
            prompt = f"{template['description']}\nUser Input: {user_input}"
            
            # Generate GPT response
            response = self.gpt_client.generate_response(prompt).strip()
            if not response:
                raise ValueError("GPT response is empty.")
            return response
        except Exception as e:
            logger.error(f"Error in generate_chat_response: {e}")
            raise RuntimeError("Failed to generate chat response.")
    
    def process_input(self, user_input: str) -> Tuple[str, Optional[int]]:
        """
        사용자 입력을 분석하여 의도를 분류합니다.
        """
        try:
            logger.info(f"Received user input: {user_input}")  # 입력 로그

            # 의도 분류
            intent_prompt = (
                f"입력: {user_input}\n"
                f"의도: (1) 향수 추천, (2) 일반 대화"
            )
            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Generated intent prompt: {intent_prompt}")  # 프롬프트 출력
            logger.info(f"Detected intent: {intent}")  # 의도 감지 결과

            if "1" in intent:
                return "recommendation", None

            return "chat", None

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to classify user intent.")

    def generate_recommendation_response(self, user_input: str) -> dict:
        try:
            logger.info(f"Processing recommendation for user input: {user_input}")

            # Validate inputs
            if not user_input:
                raise ValueError("User input is missing or invalid.")

            # Fetch perfumes based on user input
            perfumes = self.db_service.fetch_perfumes_by_user_input(user_input)
            if not perfumes:
                logger.error(f"No perfumes found for user input: {user_input}")
                raise ValueError("No perfumes found for the given user input.")

            logger.info(f"Fetched perfumes for user input {user_input}: {perfumes}")

            # 고정된 line_id 설정
            line_id = 1
            logger.info(f"Using fixed line_id: {line_id}")

            # Fetch spices for the fixed line_id
            spices = self.db_service.fetch_spices_by_line(line_id)
            if not spices:
                logger.error(f"No spices found for line_id: {line_id}")
                raise ValueError("No spices found for the given line_id.")

            # Prepare perfumes text for GPT prompt
            perfumes_text = "\n".join([
                f"{idx + 1}. {perfume['id']} ({perfume['brand']}): {perfume['description']}"
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
                    "id": f"{perfume['id']}",
                    "reason": gpt_entry.get("reason", "No specific reason provided"),
                    "situation": gpt_entry.get("situation", "No specific situation provided")
                }
                for perfume, gpt_entry in zip(perfumes, gpt_response["recommendations"])
            ]
            logger.info(f"Final recommendations: {recommendations}")

            # Return structured response with fixed line_id
            return {
                "recommendations": recommendations,
                "content": gpt_response.get("content", "No content provided"),
                "line_id": line_id
            }
        except ValueError as e:
            logger.error(f"ValueError in recommendation generation: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unhandled error in generate_recommendation_response: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations.")
