import os
from dotenv import load_dotenv
from datetime import datetime
from langgraph.graph import StateGraph
from langgraph.pregel import Channel
from typing import TypedDict, Annotated, Optional
from services.llm_service import LLMService
from services.db_service import DBService
from services.image_generation_service import ImageGenerationService
from services.llm_img_service import LLMImageService
from services.prompt_loader import PromptLoader
from models.img_llm_client import GPTClient
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class PerfumeState(TypedDict):
    user_input: Annotated[str, Channel()]
    processed_input: str  
    next_node: str
    recommendations: Optional[list]
    spices: Optional[list]
    image_path: Optional[str]
    image_description: Optional[str]
    response: Optional[str]
    line_id: Optional[int]
    translated_input: Optional[str]
    error: Optional[str]

class PerfumeService:
    def __init__(self):
        self.graph = StateGraph(state_schema=PerfumeState)

        db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }

        self.db_service = DBService(db_config)
        self.prompt_loader = PromptLoader("models/chat_prompt_template.json")
        self.gpt_client = GPTClient(self.prompt_loader)
        self.llm_service = LLMService(self.gpt_client, self.db_service, self.prompt_loader)
        self.image_service = ImageGenerationService()
        self.llm_img_service = LLMImageService(self.gpt_client)

        self.define_nodes()
        self.graph.set_entry_point("input_processor")

    def define_nodes(self):
        # Add nodes
        self.graph.add_node("input_processor", self.input_processor)
        self.graph.add_node("process_input", self.process_input)
        self.graph.add_node("recommendation_generator", self.recommendation_generator)
        self.graph.add_node("fashion_recommendation_generator", self.fashion_recommendation_generator)
        self.graph.add_node("chat_handler", self.chat_handler)
        self.graph.add_node("error_handler", self.error_handler)
        self.graph.add_node("end", lambda x: x)  # Add end node

        # Define routing function
        def route_based_on_intent(state: PerfumeState) -> str:
            if state.get("error"):
                return "error_handler"
            if state.get("processed_input") == "chat":
                return "chat_handler"
            if state.get("processed_input") == "fashion_recommendation":
                return "fashion_recommendation_generator"
            return "recommendation_generator"

        # Add conditional edges
        self.graph.add_conditional_edges(
            "process_input",
            route_based_on_intent,
            {
                "error_handler": "error_handler",
                "chat_handler": "chat_handler",
                "fashion_recommendation_generator": "fashion_recommendation_generator",
                "recommendation_generator": "recommendation_generator"
            }
        )

        # Add regular edges
        self.graph.add_edge("input_processor", "process_input")
        self.graph.add_edge("recommendation_generator", "end")  # Terminal node
        self.graph.add_edge("fashion_recommendation_generator", "end")  # Terminal node
        self.graph.add_edge("error_handler", "end")  # Terminal node
        self.graph.add_edge("chat_handler", "end")  # Terminal node

    def process_input(self, state: PerfumeState) -> PerfumeState:
        """사용자 입력을 분석하여 의도를 분류"""
        try:
            user_input = state["user_input"]  # ✅ 원본 유지
            logger.info(f"Received user input: {user_input}")

            intent_prompt = (
                f"입력: {user_input}\n"
                "다음 사용자의 의도를 분류하세요.\n"
                "의도 분류:\n"
                "(1) 향수 추천\n"
                "(2) 일반 대화\n"
                "(3) 패션 기반 향수 추천"
            )

            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Detected intent: {intent}")

            if "1" in intent:
                logger.info("💡 향수 추천 실행")
                state["processed_input"] = "recommendation"  
                state["next_node"] = "recommendation_generator"
            elif "3" in intent:
                logger.info("👕 패션 기반 향수 추천 실행")
                state["processed_input"] = "fashion_recommendation"
                state["next_node"] = "fashion_recommendation_generator"
            else:
                logger.info("💬 일반 대화 실행")
                state["processed_input"] = "chat"
                state["next_node"] = "chat_handler"

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            state["processed_input"] = "chat"
            state["next_node"] = "chat_handler"

        return state
    
    def error_handler(self, state: PerfumeState) -> PerfumeState:
        """에러 상태를 처리하고 적절한 응답을 생성하는 핸들러"""
        try:
            error_msg = state.get("error", "알 수 없는 오류가 발생했습니다")
            logger.error(f"❌ 오류 처리: {error_msg}")

            # 에러 유형에 따른 사용자 친화적 메시지 생성
            user_message = (
                "죄송합니다. "
                + (
                    "추천을 생성할 수 없습니다." if "추천" in error_msg
                    else "요청을 처리할 수 없습니다." if "처리" in error_msg
                    else "일시적인 오류가 발생했습니다."
                )
                + " 다시 시도해 주세요."
            )

            # 상태 업데이트
            state["response"] = {
                "status": "error",
                "message": user_message,
                "recommendations": [],
                "debug_info": {
                    "original_error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            }
            state["next_node"] = None  # 종료 노드로 설정

            logger.info("✅ 오류 처리 완료")
            return state

        except Exception as e:
            logger.error(f"❌ 오류 처리 중 추가 오류 발생: {e}")
            state["response"] = {
                "status": "error",
                "message": "시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                "recommendations": []
            }
            state["next_node"] = None
            return state
    
    def input_processor(self, state: PerfumeState) -> PerfumeState:
        user_input = state["user_input"]
        logger.info(f"🔍 Input: {user_input}")
        state["next_node"] = "keyword_extractor"
        return state

    def keyword_extractor(self, state: PerfumeState) -> PerfumeState:
        extracted_data = self.llm_service.extract_keywords_from_input(state["user_input"])
        logger.info(f"🔍 추출된 데이터: {extracted_data}")

        state["line_id"] = extracted_data.get("line_id", 1)
        state["next_node"] = "database_query"
        return state

    def database_query(self, state: PerfumeState) -> PerfumeState:
        line_id = state["line_id"]
        logger.info(f"✅ DB 조회 - line_id: {line_id}")

        state["spices"] = self.db_service.fetch_spices_by_line(line_id)
        state["next_node"] = "recommendation_generator"
        return state

    def recommendation_generator(self, state: PerfumeState) -> PerfumeState:
        """향수 추천 생성"""
        try:
            logger.info("🔄 향수 추천 시작")
            
            # LLM 서비스를 통한 직접 추천 생성
            try:
                response = self.llm_service.generate_recommendation_response(state["user_input"])
                
                if response and isinstance(response, dict):
                    recommendations = response.get("recommendations", [])
                    content = response.get("content", "")
                    line_id = response.get("line_id")
                    
                    if recommendations and len(recommendations) > 0:
                        logger.info(f"✅ LLM 추천 생성 완료: {len(recommendations)}개 추천됨")
                        state["recommendations"] = recommendations
                        state["response"] = {
                            "status": "success",
                            "content": content,
                            "recommendations": recommendations,
                            "line_id": line_id
                        }
                        state["next_node"] = "end"
                        return state
                        
            except Exception as e:
                logger.error(f"❌ LLM 추천 생성 실패: {e}")
                
            # DB 기반 추천 시도
            try:
                if state.get("spices"):
                    spice_ids = [spice["id"] for spice in state["spices"]]
                    filtered_perfumes = self.db_service.get_perfumes_by_middel_notes(spice_ids)
                    
                    if filtered_perfumes and len(filtered_perfumes) > 0:
                        selected_perfumes = filtered_perfumes[:3]
                        logger.info(f"✅ DB 기반 추천 완료: {len(selected_perfumes)}개 찾음")
                        
                        state["recommendations"] = selected_perfumes
                        state["response"] = {
                            "status": "success",
                            "content": "향료 기반으로 추천된 향수입니다.",
                            "recommendations": selected_perfumes,
                            "line_id": state.get("line_id", 1)
                        }
                        state["next_node"] = "end"
                        return state
                        
            except Exception as e:
                logger.error(f"❌ DB 기반 추천 실패: {e}")

            # 모든 추천 방식 실패 시
            raise ValueError("적절한 향수를 찾을 수 없습니다")

        except Exception as e:
            logger.error(f"❌ 추천 생성 오류: {e}")
            state["error"] = str(e)
            state["next_node"] = "error_handler"
            return state

    def fashion_recommendation_generator(self, state: PerfumeState) -> PerfumeState:
        """패션 기반 향수 추천 생성"""
        try:
            logger.info("🔄 향수 추천 시작")
            
            # LLM 서비스를 통한 직접 추천 생성
            try:
                response = self.llm_service.fashion_based_generate_recommendation_response(state["user_input"])
                
                if response and isinstance(response, dict):
                    recommendations = response.get("recommendations", [])
                    content = response.get("content", "")
                    line_id = response.get("line_id")
                    
                    if recommendations and len(recommendations) > 0:
                        logger.info(f"✅ LLM 추천 생성 완료: {len(recommendations)}개 추천됨")
                        state["recommendations"] = recommendations
                        state["response"] = {
                            "status": "success",
                            "content": content,
                            "recommendations": recommendations,
                            "line_id": line_id
                        }
                        state["next_node"] = "end"
                        return state
                        
            except Exception as e:
                logger.error(f"❌ LLM 추천 생성 실패: {e}")
                
            # DB 기반 추천 시도
            try:
                if state.get("spices"):
                    spice_ids = [spice["id"] for spice in state["spices"]]
                    filtered_perfumes = self.db_service.get_perfumes_by_middel_notes(spice_ids)
                    
                    if filtered_perfumes and len(filtered_perfumes) > 0:
                        selected_perfumes = filtered_perfumes[:3]
                        logger.info(f"✅ DB 기반 추천 완료: {len(selected_perfumes)}개 찾음")
                        
                        state["recommendations"] = selected_perfumes
                        state["response"] = {
                            "status": "success",
                            "content": "향료 기반으로 추천된 향수입니다.",
                            "recommendations": selected_perfumes,
                            "line_id": state.get("line_id", 1)
                        }
                        state["next_node"] = "end"
                        return state
                        
            except Exception as e:
                logger.error(f"❌ DB 기반 추천 실패: {e}")

            # 모든 추천 방식 실패 시
            raise ValueError("적절한 향수를 찾을 수 없습니다")

        except Exception as e:
            logger.error(f"❌ 추천 생성 오류: {e}")
            state["error"] = str(e)
            state["next_node"] = "error_handler"
            return state
        
    def text_translation(self, state: PerfumeState) -> PerfumeState:
        user_input = state["user_input"]

        try:
            logger.info(f"🔄 텍스트 번역 시작: {user_input}")

            translation_prompt = (
                "Translate the following Korean text to English. "
                "Ensure it is a natural and descriptive translation suitable for image generation.\n\n"
                f"Input: {user_input}\n"
                "Output:"
            )

            translated_text = self.gpt_client.generate_response(translation_prompt).strip()
            logger.info(f"✅ 번역된 텍스트: {translated_text}")

            state["translated_input"] = translated_text
            state["next_node"] = "generate_image_description"

        except Exception as e:
            logger.error(f"🚨 번역 실패: {e}")
            state["translated_input"] = "Aesthetic abstract perfume-inspired image."
            state["next_node"] = "generate_image_description"

        return state

    def image_generator(self, state: PerfumeState) -> PerfumeState:
        """추천된 향수 기반으로 이미지 생성"""
        try:
            # 추천 결과 확인
            recommendations = state.get("recommendations", [])
            if not recommendations:
                logger.warning("⚠️ 추천 결과가 없어 이미지를 생성할 수 없습니다")
                state["image_path"] = None
                state["next_node"] = "end"
                return state

            # 이미지 프롬프트 생성
            prompt_elements = []
            for rec in recommendations[:3]:  # 최대 3개 향수만 사용
                if "reason" in rec:
                    prompt_elements.append(rec["reason"])
                if "situation" in rec:
                    prompt_elements.append(rec["situation"])

            image_prompt = (
                "Create a luxurious perfume advertisement featuring: "
                f"{' '.join(prompt_elements)}. "
                "Use elegant composition and soft lighting. "
                "Style: high-end perfume photography."
            )

            logger.info(f"📸 이미지 생성 시작 - 프롬프트: {image_prompt[:100]}...")
            
            # 이미지 생성
            image_result = self.image_service.generate_image(image_prompt)
            if image_result:
                state["image_path"] = image_result.get("output_path")
                logger.info(f"✅ 이미지 생성 완료: {state['image_path']}")
            else:
                state["image_path"] = None
                logger.warning("⚠️ 이미지 생성 결과가 없습니다")

            state["next_node"] = "end"
            return state

        except Exception as e:
            logger.error(f"🚨 이미지 생성 실패: {e}")
            state["image_path"] = None
            state["next_node"] = "end"
            return state

    def chat_handler(self, state: PerfumeState) -> PerfumeState:
        try:
            user_input = state["user_input"]
            logger.info(f"💬 대화 응답 생성 시작 - 입력: {user_input}")

            state["response"] = self.llm_service.generate_chat_response(user_input)
            logger.info(f"✅ 대화 응답 생성 완료: {state['response']}")

        except Exception as e:
            logger.error(f"🚨 대화 응답 생성 실패: {e}")
            state["response"] = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."

        return state
    
    def image_description_generator(self, state: PerfumeState) -> PerfumeState:
        try:
            if "image_path" not in state or state["image_path"] is None:
                logger.warning("⚠️ 이미지 경로가 없음. 이미지 설명을 생략합니다.")
                state["image_description"] = "No image description available."
                return state

            logger.info(f"🖼 이미지 설명 생성 시작 - 이미지 경로: {state['image_path']}")
            state["image_description"] = self.llm_img_service.generate_image_description(state["image_path"])

        except Exception as e:
            logger.error(f"🚨 이미지 설명 생성 실패: {e}")
            state["image_description"] = "Failed to generate image description."

        return state

    def generate_chat_response(self, state: PerfumeState) -> PerfumeState:
        try:
            user_input = state["user_input"]
            logger.info(f"💬 대화 응답 생성 시작 - 입력: {user_input}")

            chat_prompt = (
                "당신은 향수 전문가입니다. 다음 요청에 친절하고 전문적으로 답변해주세요.\n"
                "반드시 한국어로 답변하세요.\n\n"
                f"사용자: {user_input}"
            )

            response = self.gpt_client.generate_response(chat_prompt)
            state["response"] = response.strip()
            state["next_node"] = None  # ✅ 대화 종료

        except Exception as e:
            logger.error(f"🚨 대화 응답 생성 실패: {e}")
            state["response"] = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
            state["next_node"] = None

        return state

    
    def run(self, user_input: str) -> dict:
        """그래프 실행 및 결과 반환"""
        try:
            logger.info(f"🔄 서비스 실행 시작 - 입력: {user_input}")
            
            # 초기 상태 설정
            initial_state = {
                "user_input": user_input,
                "processed_input": None,
                "next_node": None,
                "recommendations": None,
                "spices": None,
                "image_path": None,
                "image_description": None,
                "response": None,
                "line_id": None,
                "translated_input": None,
                "error": None
            }

            # 그래프 컴파일 및 실행
            compiled_graph = self.graph.compile()
            result = compiled_graph.invoke(initial_state)
            
            # 결과 검증 및 반환
            if result.get("error"):
                logger.error(f"❌ 오류 발생: {result['error']}")
                return {
                    "status": "error",
                    "message": result["error"],
                    "recommendations": []
                }
                
            logger.info("✅ 서비스 실행 완료")
            return {
                "status": "success",
                "recommendations": result.get("recommendations", []),
                "response": result.get("response"),
                "image_path": result.get("image_path")
            }

        except Exception as e:
            logger.error(f"❌ 서비스 실행 오류: {e}")
            return {
                "status": "error",
                "message": "서비스 실행 중 오류가 발생했습니다",
                "recommendations": []
            }
