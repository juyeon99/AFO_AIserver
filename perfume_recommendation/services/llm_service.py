import json
import logging
from typing import Optional, Tuple
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

    def process_input(self, user_input: str) -> Tuple[str, Optional[int]]:
        """
        사용자 입력을 분석하여 의도를 분류합니다.
        """
        try:
            logger.info(f"Received user input: {user_input}")  # 입력 로그

            # 의도 분류 프롬프트
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

    def generate_chat_response(self, user_input: str) -> str:
        """
        Generate a chat response using GPT for general conversation.
        """
        try:
            template = self.prompt_loader.get_prompt("chat")
            prompt = f"{template['description']}\nUser Input: {user_input}"
            response = self.gpt_client.generate_response(prompt).strip()
            
            if not response:
                raise ValueError("GPT response is empty.")
                
            return response
        except Exception as e:
            logger.error(f"Error in generate_chat_response: {e}")
            raise RuntimeError("Failed to generate chat response.")

    def generate_recommendation_response(self, user_input: str) -> dict:
        """
        향수를 추천하는 함수
        """
        try:
            logger.info(f"Processing recommendation for user input: {user_input}")

            # ✅ 향수 데이터 가져오기
            perfumes = self.db_service.fetch_product()
            if not perfumes:
                logger.error("No perfumes found")
                raise ValueError("No perfumes found in database")

            # ✅ 향수 정보를 GPT 프롬프트로 변환
            products_text = "\n".join([
                f"{product.get('id')}. {product.get('name_kr', 'Unknown Name')} ({product.get('brand', 'Unknown Brand')}): {product.get('content', 'No description available')}"
                for product in perfumes
            ])

            # ✅ GPT 프롬프트 템플릿 로드
            template = self.prompt_loader.get_prompt("recommendation")
            if not template or 'description' not in template:
                raise ValueError("Invalid recommendation prompt template")

            # ✅ GPT 프롬프트 생성
            final_prompt = (
                f"{template['description']}\n"
                f"Products:\n{products_text}\n"
                "You must respond with a valid JSON object containing:\n"
                "{\n"
                '  "recommendations": [\n'
                '    {"reason": "...", "situation": "..."}\n'
                "  ],\n"
                '  "content": "summary of common feeling"\n'
                "}\n"
                "Ensure the response is a properly formatted JSON object."
            )

            # ✅ GPT 응답 생성 및 파싱
            response_text = self.gpt_client.generate_response(final_prompt)
            if not response_text:
                raise ValueError("Empty GPT response")

            # ✅ JSON 구조 추출
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No valid JSON structure found in response")
                    
                json_str = response_text[start_idx:end_idx + 1]
                gpt_response = json.loads(json_str)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {response_text}")
                logger.error(f"JSON error: {e}")
                # Fallback response
                gpt_response = {
                    "recommendations": [],
                    "content": "향수들의 공통적인 특징을 분석할 수 없습니다."
                }

            # ✅ 응답 검증 및 기본값 설정
            gpt_response.setdefault('recommendations', [])
            gpt_response.setdefault('content', "향수들의 공통적인 특징을 분석할 수 없습니다.")

            # ✅ 최종 응답 생성
            recommendations = [
                {
                    "id": str(perfumes[idx]['id']),
                    "reason": rec.get('reason', "No reason provided"),
                    "situation": rec.get('situation', "No situation provided")
                }
                for idx, rec in enumerate(gpt_response.get('recommendations', [])[:3])
                if idx < len(perfumes)
            ]

            return {
                "recommendations": recommendations,
                "content": gpt_response['content'],
                "line_id": perfumes[0].get('line_id', None) if perfumes else None
            }

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations")
