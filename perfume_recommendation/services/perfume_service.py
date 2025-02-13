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
    
    """
    향수 추천 서비스의 상태를 관리하는 타입 정의
    
    Attributes:
        user_input (str): 사용자의 입력 텍스트
            - Channel()을 통해 상태 그래프에서 데이터 흐름 관리
        processed_input (str): 처리된 입력 텍스트
            - 의도 분류 결과 저장 (recommendation, chat 등)
        next_node (str): 다음 실행할 노드의 이름
            - 그래프 흐름 제어를 위한 다음 노드 지정
        recommendations (list): 추천된 향수 목록
            - LLM 또는 DB에서 생성된 향수 추천 목록
        spices (list): 추출된 향료 정보 목록
            - 향 계열에 따른 향료 정보
        image_path (str): 생성된 이미지 경로
            - 이미지 생성 결과물 저장 경로
        image_description (str): 이미지 설명
            - 생성된 이미지에 대한 설명 텍스트
        response (str): 응답 메시지
            - 최종 사용자 응답 데이터
        line_id (int): 향 계열 ID
            - 향수의 계열 분류 ID
        translated_input (str): 번역된 입력 텍스트
            - 이미지 생성을 위한 영문 번역 텍스트
        error (str): 오류 메시지
            - 처리 중 발생한 오류 정보
    """
    
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
        self.graph.add_node("recommendation_type_classifier", self.recommendation_type_classifier)  # 추가
        self.graph.add_node("recommendation_generator", self.recommendation_generator)
        self.graph.add_node("fashion_recommendation_generator", self.fashion_recommendation_generator)
        self.graph.add_node("chat_handler", self.chat_handler)
        self.graph.add_node("error_handler", self.error_handler)
        self.graph.add_node("end", lambda x: x)

        # router Function
        def route_based_on_intent(state: PerfumeState) -> str:
            if state.get("error"):
                return "error_handler"
            if state.get("processed_input") == "chat":
                return "chat_handler"
            if state.get("processed_input") == "fashion_recommendation":
                return "fashion_recommendation_generator"
            if state.get("processed_input") == "general_recommendation":
                return "recommendation_generator"
            return "recommendation_type_classifier"  # 향수 추천이면 추가 분류로 이동

        # if_rogic
        self.graph.add_conditional_edges(
            "process_input",
            route_based_on_intent,
            {
                "error_handler": "error_handler",
                "chat_handler": "chat_handler",
                "recommendation_type_classifier": "recommendation_type_classifier",  # 추가된 노드
                "fashion_recommendation_generator": "fashion_recommendation_generator",
                "recommendation_generator": "recommendation_generator",
            }
        )

        # if_router_type
        def route_recommendation_type(state: PerfumeState) -> str:
            if state.get("processed_input") == "fashion_recommendation":
                return "fashion_recommendation_generator"
            return "recommendation_generator"

        self.graph.add_conditional_edges(
            "recommendation_type_classifier",
            route_recommendation_type,
            {
                "fashion_recommendation_generator": "fashion_recommendation_generator",
                "recommendation_generator": "recommendation_generator",
            }
        )

        # Add_edge
        self.graph.add_edge("input_processor", "process_input")
        self.graph.add_edge("recommendation_generator", "end")
        self.graph.add_edge("fashion_recommendation_generator", "end")
        self.graph.add_edge("error_handler", "end")
        self.graph.add_edge("chat_handler", "end")

    def process_input(self, state: PerfumeState) -> PerfumeState:
        """사용자 입력을 분석하여 의도를 분류"""
        try:
            user_input = state["user_input"]  
            logger.info(f"Received user input: {user_input}")

            intent_prompt = (
                f"입력: {user_input}\n"
                f"다음 사용자의 의도를 분류하세요.\n\n"
                f"일반적인 키워드라고 볼 수 없는 향수 추천은 (2) 일반 대화로 분류해야 합니다.\n\n"
                f"예시) user_input = 나 오늘 기분이 너무 우울해. 그래서 이런 기분을 떨쳐낼 수 있는 플로럴 계열의 향수를 추천해줘 (1) 향수 추천 \n"
                f"user_input = 나는 오늘 데이트를 하러가는데 추천해줄 만한 향수가 있을까? (1) 향수 추천 \n"
                f"예시) user_input = 향수를 추천받고 싶은데 뭐 좋은 거 있어? (2) 일반 대화\n"
                f"예시) user_input = 향수를 추천해주세요. 라면 (2) 일반 대화로 분류해야 합니다.\n\n"
                f"의도: (1) 향수 추천, (2) 일반 대화"
            )

            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Detected intent: {intent}")

            if "1" in intent:
                logger.info("💡 향수 추천 실행")
                state["processed_input"] = "recommendation"  
                state["next_node"] = "recommendation_type_classifier"  # 추천 유형 분류로 이동
            else:
                logger.info("💬 일반 대화 실행")
                state["processed_input"] = "chat"
                state["next_node"] = "chat_handler"
        
        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            state["processed_input"] = "chat"
            state["next_node"] = "chat_handler"

        return state

    def recommendation_type_classifier(self, state: PerfumeState) -> PerfumeState:
        """향수 추천 유형을 추가적으로 분류 (패션 추천 vs 일반 추천)"""
        try:
            user_input = state["user_input"]
            logger.info(f"향수 추천 유형 분류 시작 - 입력: {user_input}")

            type_prompt = (
                f"입력: {user_input}\n"
                f"향수 추천을 패션 기반 추천과 일반 추천으로 나누세요.\n\n"
                f"예시) user_input = 나는 오늘 수트를 입었는데 어울리는 향수가 필요해 (3) 패션 추천\n"
                f"예시) user_input = 상큼한 향이 나는 향수를 추천해줘 (4) 일반 추천\n\n"
                f"의도: (3) 패션 추천, (4) 일반 추천"
            )

            recommendation_type = self.gpt_client.generate_response(type_prompt).strip()
            logger.info(f"Detected recommendation type: {recommendation_type}")

            if "3" in recommendation_type:
                logger.info("👕 패션 기반 향수 추천 실행")
                state["processed_input"] = "fashion_recommendation"
                state["next_node"] = "fashion_recommendation_generator"
            else:
                logger.info("✨ 일반 향수 추천 실행")
                state["processed_input"] = "general_recommendation"
                state["next_node"] = "recommendation_generator"
        
        except Exception as e:
            logger.error(f"Error processing recommendation type '{user_input}': {e}")
            state["processed_input"] = "general_recommendation"
            state["next_node"] = "recommendation_generator"
        
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

                    logger.info("✅ LLM 추천 생성 완료")

                    state["response"] = {
                        "status": "success",
                        "mode": "recommendation",
                        "recommendations": recommendations,
                        "content": content,
                        "line_id": line_id
                    }

                    # 이미지 생성 시도
                    try:
                        image_state = self.image_generator(state)
                        state["image_path"] = image_state.get("image_path")
                        if state["image_path"]:
                            logger.info(f"✅ 이미지 생성 성공: {state['image_path']}")
                            state["response"]["image_path"] = state["image_path"]
                        else:
                            logger.warning("⚠️ 이미지 생성 실패")
                    except Exception as img_err:
                        logger.error(f"❌ 이미지 생성 오류: {img_err}")
                        state["image_path"] = None

                    state["next_node"] = "end"
                    return state

            except Exception as e:
                logger.error(f"❌ LLM 추천 생성 실패: {e}")

            # DB 기반 추천 시도
            try:
                if state.get("spices"):
                    spice_ids = [spice["id"] for spice in state["spices"]]
                    filtered_perfumes = self.db_service.get_perfumes_by_middel_notes(spice_ids)

                    if filtered_perfumes:
                        logger.info(f"✅ DB 기반 추천 완료: {len(filtered_perfumes)}개 찾음")

                        state["response"] = {
                            "status": "success",
                            "mode": "recommendation",
                            "recommendations": filtered_perfumes,
                            "content": "향료 기반으로 추천된 향수입니다.",
                            "line_id": state.get("line_id", 1)
                        }

                        # 이미지 생성 시도
                        try:
                            image_state = self.image_generator(state)
                            state["image_path"] = image_state.get("image_path")
                            if state["image_path"]:
                                logger.info(f"✅ 이미지 생성 성공: {state['image_path']}")
                                state["response"]["image_path"] = state["image_path"]
                            else:
                                logger.warning("⚠️ 이미지 생성 실패")
                        except Exception as img_err:
                            logger.error(f"❌ 이미지 생성 오류: {img_err}")
                            state["image_path"] = None

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

                    logger.info("✅ LLM 추천 생성 완료")

                    state["response"] = {
                        "status": "success",
                        "mode": "recommendation",
                        "recommendation": recommendations,
                        "content": content,
                        "line_id": line_id
                    }

                    # 이미지 생성 시도
                    try:
                        image_state = self.image_generator(state)
                        state["image_path"] = image_state.get("image_path")
                        if state["image_path"]:
                            logger.info(f"✅ 이미지 생성 성공: {state['image_path']}")
                            state["response"]["image_path"] = state["image_path"]
                        else:
                            logger.warning("⚠️ 이미지 생성 실패")
                    except Exception as img_err:
                        logger.error(f"❌ 이미지 생성 오류: {img_err}")
                        state["image_path"] = None

                    state["next_node"] = "end"
                    return state

            except Exception as e:
                logger.error(f"❌ LLM 추천 생성 실패: {e}")

            # DB 기반 추천 시도
            try:
                if state.get("spices"):
                    spice_ids = [spice["id"] for spice in state["spices"]]
                    filtered_perfumes = self.db_service.get_perfumes_by_middel_notes(spice_ids)

                    if filtered_perfumes:
                        logger.info(f"✅ DB 기반 추천 완료: {len(filtered_perfumes)}개 찾음")

                        state["response"] = {
                            "status": "success",
                            "mode": "fashion_recommendation",
                            "recommendation": filtered_perfumes,
                            "content": "향료 기반으로 추천된 향수입니다.",
                            "line_id": state.get("line_id", 1)
                        }

                        # 이미지 생성 시도
                        try:
                            image_state = self.image_generator(state)
                            state["image_path"] = image_state.get("image_path")
                            if state["image_path"]:
                                logger.info(f"✅ 이미지 생성 성공: {state['image_path']}")
                                state["response"]["image_path"] = state["image_path"]
                            else:
                                logger.warning("⚠️ 이미지 생성 실패")
                        except Exception as img_err:
                            logger.error(f"❌ 이미지 생성 오류: {img_err}")
                            state["image_path"] = None

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

            translated_text = self.llm_img_service.generate_image_description(translation_prompt).strip()
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
            # ✅ response 객체 내부의 "recommendations" 및 "content" 안전하게 검증
            response = state.get("response") or {}  
            recommendations = response.get("recommendations") or []  
            content = response.get("content", "")

            if not recommendations:
                logger.warning("⚠️ response 객체 내 추천 결과가 없어 이미지를 생성할 수 없습니다")
                response["image_path"] = ""
                state["next_node"] = "end"
                return state

            # 이미지 프롬프트 생성
            prompt_parts = []

            # Content 번역
            try:
                if content:
                    content_translation_prompt = (
                        "Translate the following Korean text to English, maintaining the professional tone:\n\n"
                        f"Text: {content}\n"
                        "Translation:"
                    )
                    translated_content = self.gpt_client.generate_response(content_translation_prompt).strip()
                    prompt_parts.append(translated_content)
                    logger.info("✅ Content 번역 완료")

                # 각 추천 항목에 대해 영어로 번역
                translated_recommendations = []
                for rec in recommendations[:3]:  # 최대 3개만 처리
                    # 번역이 필요한 텍스트 구성
                    translation_text = (
                        f"Name: {rec.get('name', '')}\n"
                        f"Brand: {rec.get('brand', '')}\n"
                        f"Reason: {rec.get('reason', '')}\n"
                        f"Situation: {rec.get('situation', '')}"
                    )
                    
                    # text_translation을 통한 번역
                    translation_state = {"user_input": translation_text}
                    translated_state = self.text_translation(translation_state)
                    translated_text = translated_state.get("translated_input", "")
                    
                    # 번역된 텍스트 파싱
                    translated_parts = translated_text.split("\n")
                    translated_rec = {
                        "name": translated_parts[0].replace("Name: ", "").strip(),
                        "brand": translated_parts[1].replace("Brand: ", "").strip(),
                        "reason": translated_parts[2].replace("Reason: ", "").strip(),
                        "situation": translated_parts[3].replace("Situation: ", "").strip()
                    }
                    translated_recommendations.append(translated_rec)

                # 번역된 정보로 프롬프트 구성
                for rec in translated_recommendations:
                    if rec['reason']:
                        prompt_parts.append(rec['reason'])
                    if rec['situation']:
                        atmosphere = rec['situation'].split(',')[0]
                        prompt_parts.append(atmosphere)

                logger.info("✅ 텍스트 번역 완료")

            except Exception as trans_err:
                logger.error(f"❌ 번역 실패: {trans_err}")
                # 번역 실패 시 기본 프롬프트 사용
                prompt_parts = ["Elegant and sophisticated fragrance ambiance"
                                "A refined and luxurious scent experience"
                                "Aesthetic and harmonious fragrance composition"
                                "An artistic representation of exquisite aromas"
                                "A sensory journey of delicate and captivating scents"]

            # 이미지 프롬프트 구성
            image_prompt = (
            "Create a professional Sentique advertisement image that immerses the viewer in a luxurious and sensory fragrance experience. The image should evoke an elegant and enchanting atmosphere, focusing on the essence of scent without displaying a perfume bottle.\n\n"
            "Characteristics:\n"
            "- A delicate interplay of light and shadow, enhancing depth and mystery\n"
            "- Ethereal, dreamlike mist that conveys the diffusion of fragrance in the air\n"
            "- A harmonious blend of soft pastels or deep, moody hues to reflect various scent profiles\n"
            "- Abstract visual storytelling that hints at floral, woody, citrus, or oriental fragrance families\n"
            "- Intricate textures, such as flowing silk, delicate petals, or aged parchment, to symbolize complexity and richness of the scent\n"
            "- A refined composition that exudes elegance, avoiding direct product representation\n"
            "- Motion elements like floating particles, swirling essence, or diffused reflections to create an immersive ambiance\n\n"
            f"{''.join(prompt_parts)}"
            "Requirements:\n"
            "- Cinematic lighting with a soft glow to enhance warmth and depth\n"
            "- Artistic and sophisticated styling, ensuring an upscale, luxurious feel\n"
            "- Emphasize the feeling of the scent rather than describing the perfume bottle clearly. The perfume bottle does not appear.\n"
            "- Professional color grading to maintain visual harmony and depth\n"
            )
            logger.info(f"📸 이미지 생성 시작\n프롬프트: {image_prompt}")

            # ✅ 이미지 저장 경로 지정 (generated_images 폴더)
            save_directory = "generated_images"
            os.makedirs(save_directory, exist_ok=True)  # 폴더가 없으면 생성

            try:
                image_result = self.image_service.generate_image(image_prompt)

                if not image_result:
                    raise ValueError("❌ 이미지 생성 결과가 비어있습니다")

                if not isinstance(image_result, dict):
                    raise ValueError(f"❌ 잘못된 이미지 결과 형식: {type(image_result)}")

                raw_output_path = image_result.get("output_path")
                if not raw_output_path:
                    raise ValueError("❌ 이미지 경로가 없습니다")

                # ✅ 저장 경로를 `generated_images/` 폴더로 변경
                filename = os.path.basename(raw_output_path)
                output_path = os.path.join(save_directory, filename)

                # ✅ 파일을 `generated_images/` 폴더로 이동
                if os.path.exists(raw_output_path):
                    os.rename(raw_output_path, output_path)

                # ✅ `response["image_path"]`에 최종 경로 설정
                response["image_path"] = output_path
                logger.info(f"✅ 이미지 생성 완료: {output_path}")

            except Exception as img_err:
                logger.error(f"🚨 이미지 생성 실패: {img_err}")
                response["image_path"] = "failed"  # ✅ 실패 시 "failed"로 설정

            state["next_node"] = "end"
            return state

        except Exception as e:
            logger.error(f"❌ 이미지 생성 오류: {e}")
            state["error"] = str(e)
            state["next_node"] = "error_handler"
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
            state["content"] = response.strip()
            state["next_node"] = None  # ✅ 대화 종료

        except Exception as e:
            logger.error(f"🚨 대화 응답 생성 실패: {e}")
            state["content"] = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
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
                "response": result.get("response"),
            }

        except Exception as e:
            logger.error(f"❌ 서비스 실행 오류: {e}")
            return {
                "status": "error",
                "message": "서비스 실행 중 오류가 발생했습니다",
                "recommendations": []
            }
