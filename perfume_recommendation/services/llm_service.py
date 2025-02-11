import json
import logging
from typing import Optional, Tuple
from models.img_llm_client import GPTClient
from .db_service import DBService
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
                f"다음 사용자의 의도를 분류하세요.\n\n"
                f"일반적인 키워드라고 볼 수 없는 향수 추천은 (2) 일반 대화로 분류해야 합니다.\n\n"
                f"예시) user_input = 나 오늘 기분이 너무 우울해. 그래서 이런 기분을 떨쳐낼 수 있는 플로럴 계열의 향수를 추천해줘 (1) 향수 추천 \n"
                f"예시) user_input = 향수를 추천받고 싶은데 뭐 좋은 거 있어? (2) 일반 대화\n"
                f"예시) user_input = 향수를 추천해주세요. 라면 (2) 일반 대화로 분류해야 합니다.\n\n"
                f"의도: (1) 향수 추천, (2) 일반 대화, (3) 패션 향수 추천"
            )

            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Detected intent: {intent}")  # 의도 감지 결과

            if "1" in intent:
                logger.info("💡 일반 향수 추천 실행")
                return "recommendation", self.generate_recommendation_response(user_input)

            if "3" in intent:
                logger.info("👕 패션 기반 향수 추천 실행 (mode는 recommendation 유지)")
                return "recommendation", self.fashion_based_generate_recommendation_response(user_input)

            return "chat", self.generate_chat_response(user_input)

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to classify user intent.")

    def extract_keywords_from_input(self, user_input: str) -> dict:
        """사용자 입력에서 계열과 브랜드를 분석하고 계열 ID와 브랜드 리스트를 반환하는 함수"""
        try:
            logger.info(f"🔍 입력된 텍스트에서 향 계열과 브랜드 분석 시작: {user_input}")

            # 1. DB에서 계열 및 브랜드 데이터 가져오기
            line_data = self.db_service.fetch_line_data()
            line_mapping = {line["name"]: line["id"] for line in line_data}
            brand_list = self.db_service.fetch_brands()

            fashion_to_line_mapping = {
                # 캐주얼 스타일
                "캐주얼": "Fruity",
                "댄디 캐주얼": "Woody",  # 댄디하면서도 세련된 스타일
                "아메카지": "Green",  # 내추럴하면서 빈티지한 느낌  

                # 클래식 & 포멀 스타일
                "클래식": "Woody",
                "비즈니스 포멀": "Musk",  # 정장 착장에 어울리는 차분한 향
                "비즈니스 캐주얼": "Citrus",  # 가벼운 포멀 룩에 잘 맞는 시원한 향
                "젠틀한 스타일": "Powdery",  # 부드러운 분위기를 주는 Powdery 향  

                # 스트릿 & 유니크 스타일
                "스트릿": "스파이시",
                "테크웨어": "아로마틱",  # SF적이고 미래적인 느낌의 패션과 어울림
                "고프코어": "Green",  # 등산 및 아웃도어 느낌의 스타일과 자연스러운 향
                "펑크 스타일": "Tobacco Leather",  # 강렬한 락 & 펑크 무드  

                # 스포티 & 액티브 스타일
                "스포티": "Citrus",
                "러너 스타일": "Aquatic",  # 활동적이고 신선한 느낌  
                "테니스 룩": "Fougere",  # 클래식하면서도 깨끗한 향  

                # 빈티지 & 감성적인 스타일
                "빈티지": "Oriental",
                "로맨틱 스타일": "Floral",  # 부드럽고 달콤한 분위기의 스타일  
                "보헤미안": "Musk",  # 자연스럽고 몽환적인 분위기  
                "레트로 패션": "Aldehyde",  # 70~80년대 스타일과 어울리는 클래식한 향  

                # 모던 & 미니멀 스타일
                "모던": "Woody",
                "미니멀": "Powdery",  # 깨끗하고 단정한 분위기  
                "올 블랙 룩": "Tobacco Leather",  # 강렬하면서 시크한 무드  
                "화이트 톤 스타일": "Musk",  # 깨끗하고 부드러운 느낌  

                # 독특한 컨셉 스타일
                "아방가르드": "Tobacco Leather",  # 예술적인 스타일과 어울리는 가죽 향  
                "고딕 스타일": "Oriental",  # 다크하면서 무게감 있는 향  
                "코스프레": "Gourmand",  # 달콤하면서 개성 강한 스타일  
            }
            
            # 2. GPT를 이용해 입력에서 향 계열과 브랜드 추출
            keywords_prompt = (
                "다음은 향수 추천 요청입니다. 사용자의 입력에서 향 계열과 브랜드명을 추출하세요.\n"
                f"향 계열 목록: {', '.join(line_mapping.keys())}\n"
                f"브랜드 목록: {', '.join(brand_list)}\n\n"
                f"사용자 입력: {user_input}\n\n"
                "추가 규칙: 만약 사용자의 입력이 패션 스타일에 대한 설명이라면, 다음 패션 스타일과 어울리는 향 계열을 사용하세요.\n"
                f"{json.dumps(fashion_to_line_mapping, ensure_ascii=False, indent=2)}\n\n"
                "출력 형식은 JSON이어야 합니다:\n"
                "{\n"
                '  "line": "우디",\n'
                '  "brands": ["샤넬", "딥티크"]\n'
                "}"
            )

            response_text = self.gpt_client.generate_response(keywords_prompt).strip()
            logger.info(f"🤖 GPT 응답: {response_text}")

            # 3. JSON 변환
            try:
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()

                parsed_response = json.loads(response_text)
                extracted_line_name = parsed_response.get("line", "").strip()
                extracted_brands = parsed_response.get("brands", [])

                # 4. 계열 ID 찾기
                line_id = line_mapping.get(extracted_line_name)
                if not line_id:
                    raise ValueError(f"❌ '{extracted_line_name}' 계열이 존재하지 않습니다.")

                logger.info(f"✅ 계열 ID: {line_id}, 브랜드: {extracted_brands}")

                return {
                    "line_id": line_id,
                    "brands": extracted_brands
                }

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON 파싱 오류: {e}")
                logger.error(f"📄 GPT 응답 원본: {response_text}")
                raise ValueError("❌ JSON 파싱 실패")

        except Exception as e:
            logger.error(f"❌ 키워드 추출 오류: {e}")
            raise ValueError(f"❌ 키워드 추출 실패: {str(e)}")


    def generate_chat_response(self, user_input: str) -> str:
        """일반 대화 응답을 생성하는 함수"""
        try:
            logger.info(f"💬 대화 응답 생성 시작 - 입력: {user_input}")

            # 1. 프롬프트 생성
            template = self.prompt_loader.get_prompt("chat")
            chat_prompt = (
                f"{template['description']}\n"
                f"{template['rules']}\n"
                f"{template['example_prompt']}\n"
                "당신은 향수 전문가입니다. 다음 요청에 친절하고 전문적으로 답변해주세요.\n"
                "단, 향수 추천은 하지만 일반적인 정보만 제공하고 , 반드시 한국어로 답변하세요.\n\n"
                f"사용자: {user_input}"
            )
            logger.debug(f"📝 생성된 프롬프트:\n{chat_prompt}")

            # 2. GPT 응답 요청
            logger.info("🤖 GPT 응답 요청")
            response = self.gpt_client.generate_response(chat_prompt)
            
            if not response:
                logger.error("❌ GPT 응답이 비어있음")
                raise ValueError("응답 생성 실패")

            logger.info("✅ 응답 생성 완료")
            return response.strip()

        except Exception as e:
            logger.error(f"❌ 대화 응답 생성 오류: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"대화 응답 생성 실패: {str(e)}"
        )

    def generate_recommendation_response(self, user_input: str) -> dict:
        """meddel note를 포함한 향수 추천"""
        try:
            logger.info(f"🔄 추천 처리 시작 - 입력: {user_input}")

            # 1. 키워드 추출 
            logger.info("🔍 키워드 추출 시작")
            extracted_data = self.extract_keywords_from_input(user_input)
            line_id = extracted_data["line_id"]
            brand_filters = extracted_data["brands"]
            logger.info(f"✅ 추출된 키워드 - 계열ID: {line_id}, 브랜드: {brand_filters}")

            # 2. 향료 ID 조회
            logger.info(f"🔍 계열 {line_id}의 향료 조회")
            spice_data = self.db_service.fetch_spices_by_line(line_id)
            spice_ids = [spice["id"] for spice in spice_data]

            if not spice_ids:
                logger.error(f"❌ 계열 {line_id}에 대한 향료 없음")
                raise HTTPException(status_code=404, detail="해당 계열에 맞는 향료를 찾을 수 없습니다")
            
            logger.info(f"✅ 향료 ID 목록: {spice_ids}")

            # 3. 향수 필터링
            logger.info("🔍 향수 필터링 시작")
            filtered_perfumes = self.db_service.get_perfumes_by_middel_notes(spice_ids)
            logger.debug(f"📋 미들노트 기준 필터링: {len(filtered_perfumes)}개")

            if brand_filters:
                filtered_perfumes = [p for p in filtered_perfumes if p["brand"] in brand_filters]
                logger.debug(f"📋 브랜드 필터링 후: {len(filtered_perfumes)}개")

            if not filtered_perfumes:
                logger.error("❌ 필터링 결과 없음")
                raise HTTPException(status_code=404, detail="조건에 맞는 향수를 찾을 수 없습니다")

            # 4. GPT 프롬프트 생성
            products_text = "\n".join([
                f"{p['id']}. {p['name_kr']} ({p['brand']}): {p.get('main_accord', '향 정보 없음')}"
                for p in filtered_perfumes[:50]  # 최대 10개로 제한
            ])

            template = self.prompt_loader.get_prompt("recommendation")
            names_prompt = (
                f"{template['description']}\n"
                f"{template['rules']}"
                f"사용자 요청: {user_input}\n"
                f"추출된 키워드: {products_text}\n"
                f"향수의 브랜드 이름은 들어가지 않은 이름만 최대 3개 추천해주세요.\n\n"
                f"- contetn: 추천 이유와 사용 상황과 향수들의 공통적인 느낌 함께 적어주세요.\n\n"
                f"- line_id: 추천된 향수들의 공통적인 계열 아이디를 작성합니다.\n"
                "아래 JSON 형식으로만 응답하세요:\n"
                "```json\n"
                "{\n"
                '  "recommendations": [\n'
                '    {\n'
                '      "name": "블랑쉬 오 드 퍼퓸",\n'
                '      "reason": "깨끗한 머스크와 은은한 백합이 어우러져, 갓 세탁한 새하얀 리넨처럼 부드럽고 신선한 느낌을 선사. 피부에 밀착되는 듯한 가벼운 향이 오래 지속되며, 자연스럽고 단정한 분위기를 연출함.",\n'
                '      "situation": "아침 샤워 후 상쾌한 기분을 유지하고 싶을 때, 오피스에서 단정하면서도 은은한 존재감을 남기고 싶을 때"\n'
                '    },\n'
                '    {\n'
                '      "name": "실버 마운틴 워터 오 드 퍼퓸",\n'
                '      "reason": "상큼한 시트러스와 신선한 그린 티 노트가 조화를 이루며, 알프스의 깨끗한 샘물을 연상시키는 맑고 청량한 느낌을 줌. 우디한 베이스가 잔잔하게 남아 차분한 매력을 더함.",\n'
                '      "situation": "운동 후 땀을 씻어내고 개운한 느낌을 유지하고 싶을 때, 더운 여름날 시원하고 깨끗한 인상을 주고 싶을 때"\n'
                '    },\n'
                '    {\n'
                '      "name": "재즈 클럽 오 드 뚜왈렛",\n'
                '      "reason": "달콤한 럼과 부드러운 바닐라가 타바코의 스모키함과 어우러져, 클래식한 재즈 바에서 오래된 가죽 소파에 앉아 칵테일을 마시는 듯한 분위기를 연출. 깊고 따뜻한 향이 감각적인 무드를 더함.",\n'
                '      "situation": "여유로운 저녁 시간, 칵테일 바나 조용한 라운지에서 세련된 분위기를 연출하고 싶을 때, 가을과 겨울철 따뜻하고 매혹적인 향을 원할 때"\n'
                '    }\n'
                '  ]\n'
                '}\n'
                'content: "깨끗한 리넨의 산뜻함, 신선한 자연의 청량감, 그리고 부드러운 따뜻함이 조화롭게 어우러진 세련되고 감각적인 향입니다.."\n'
                'line_id: 14\n'
                "```"
            )

            try:
                logger.info("🔄 향수 추천 처리 시작")
                
                # 1. GPT 응답 받기
                logger.info("🤖 GPT 응답 요청")
                response_text = self.gpt_client.generate_response(names_prompt)
                logger.debug(f"📝 GPT 원본 응답:\n{response_text}")

                # 2. JSON 파싱
                try:
                    # 마크다운 코드 블록 제거
                    if '```' in response_text:
                        parts = response_text.split('```')
                        for part in parts:
                            if '{' in part and '}' in part:
                                response_text = part.strip()
                                if response_text.startswith('json'):
                                    response_text = response_text[4:].strip()
                                break

                    # JSON 구조 추출
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if (start_idx == -1 or end_idx <= start_idx):
                        raise ValueError("JSON 구조를 찾을 수 없습니다")
                        
                    json_str = response_text[start_idx:end_idx]
                    logger.debug(f"📋 추출된 JSON:\n{json_str}")
                    
                    gpt_response = json.loads(json_str)
                    logger.info("✅ JSON 파싱 성공")

                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON 파싱 오류: {e}")
                    logger.error(f"📄 파싱 시도한 텍스트:\n{json_str if 'json_str' in locals() else 'None'}")
                    raise ValueError("JSON 파싱 실패")

                # 3. 추천 목록 생성
                recommendations = []
                for rec in gpt_response.get("recommendations", []):
                    matched_perfume = next(
                        (p for p in filtered_perfumes if p["name_kr"] == rec["name"]), 
                        None
                    )

                    if matched_perfume:
                        recommendations.append({
                            "id": matched_perfume["id"],
                            "name": matched_perfume["name_kr"], 
                            "brand": matched_perfume["brand"],
                            "reason": rec.get("reason", "추천 이유 없음"),
                            "situation": rec.get("situation", "사용 상황 없음")
                        })

                if not recommendations:
                    logger.error("❌ 유효한 추천 결과 없음")
                    raise ValueError("유효한 추천 결과가 없습니다")

                # 4. 공통 line_id 찾기
                common_line_id = self.get_common_line_id(recommendations)
                logger.info(f"✅ 공통 계열 ID: {common_line_id}")

                return {
                    "recommendations": recommendations,
                    "content": gpt_response.get("content", "추천 분석 실패"),
                    "line_id": common_line_id
                }

            except ValueError as ve:
                logger.error(f"❌ 추천 처리 오류: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                logger.error(f"❌ 예상치 못한 오류: {e}")
                raise HTTPException(status_code=500, detail="추천 생성 실패")

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            raise HTTPException(status_code=500, detail="추천 JSON 파싱 실패")
        except Exception as e:
            logger.error(f"추천 생성 오류: {str(e)}")
            raise HTTPException(status_code=500, detail="추천 생성 실패")

    def get_common_line_id(self, recommendations: list) -> int:
        """추천된 향수들의 공통 계열 ID를 찾는 함수"""
        try:
                logger.info("🔍 GPT를 이용한 공통 계열 ID 검색 시작")

                if not recommendations:
                    logger.warning("⚠️ 추천 목록이 비어 있음") 
                    return 1

                # 1. DB에서 line 데이터 가져오기
                line_data = self.db_service.fetch_line_data()
                if not line_data:
                    logger.error("❌ 계열 데이터를 찾을 수 없음")
                    return 1
                    
                # 향수 계열 정보 생성
                line_info = "\n".join([
                    f"{line['id']}: {line['name']} - {line.get('description', '설명 없음')}"
                    for line in line_data
                ])

                # 2. 향수 목록 생성
                perfume_list = ", ".join([
                    f"{rec['name']} ({rec['id']})" 
                    for rec in recommendations
                ])
                logger.debug(f"📋 분석할 향수 목록: {perfume_list}")

                # 3. GPT 프롬프트 생성 
                prompt = (
                    f"다음 향수 목록을 보고 가장 적합한 계열 ID를 선택해주세요.\n\n"
                    f"향수 목록:\n{perfume_list}\n\n"
                    f"계열 정보:\n{line_info}\n\n"
                    "다음 JSON 형식으로만 응답하세요:\n"
                    "{\n"
                    '  "line_id": 선택한_ID\n'
                    "}"
                )

                # 4. GPT 요청
                logger.info("🤖 GPT 응답 요청") 
                response = self.gpt_client.generate_response(prompt)
                logger.debug(f"📝 GPT 응답:\n{response}")

                # 5. JSON 파싱 및 검증
                try:
                    clean_response = response.strip()
                    
                    # 마크다운 블록 제거
                    if '```' in clean_response:
                        parts = clean_response.split('```')
                        for part in parts:
                            if '{' in part and '}' in part:
                                clean_response = part.strip()
                                if clean_response.startswith('json'):
                                    clean_response = clean_response[4:].strip()
                                break

                    # JSON 추출
                    json_str = clean_response[
                        clean_response.find('{'):
                        clean_response.rfind('}')+1
                    ]
                    
                    response_data = json.loads(json_str)
                    line_id = response_data.get('line_id')

                    # line_id 검증
                    valid_ids = {line['id'] for line in line_data}
                    if not isinstance(line_id, int) or line_id not in valid_ids:
                        raise ValueError(f"유효하지 않은 line_id: {line_id}")

                    logger.info(f"✅ 공통 계열 ID 찾음: {line_id}")
                    return line_id

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"❌ JSON 파싱/검증 오류: {e}")
                    return 1

        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {e}")
            return 1
        
    def fashion_based_generate_recommendation_response(self, user_input: str) -> dict:
        """meddel note를 포함한 향수 추천"""
        try:
            logger.info(f"🔄 추천 처리 시작 - 입력: {user_input}")

            # 1. 키워드 추출 
            logger.info("🔍 키워드 추출 시작")
            extracted_data = self.extract_keywords_from_input(user_input)
            line_id = extracted_data["line_id"]
            brand_filters = extracted_data["brands"]
            logger.info(f"✅ 추출된 키워드 - 계열ID: {line_id}, 브랜드: {brand_filters}")

            # 2. 향료 ID 조회
            logger.info(f"🔍 계열 {line_id}의 향료 조회")
            spice_data = self.db_service.fetch_spices_by_line(line_id)
            spice_ids = [spice["id"] for spice in spice_data]

            if not spice_ids:
                logger.error(f"❌ 계열 {line_id}에 대한 향료 없음")
                raise HTTPException(status_code=404, detail="해당 계열에 맞는 향료를 찾을 수 없습니다")
            
            logger.info(f"✅ 향료 ID 목록: {spice_ids}")

            # 3. 향수 필터링
            logger.info("🔍 향수 필터링 시작")
            filtered_perfumes = self.db_service.get_perfumes_by_middel_notes(spice_ids)
            logger.debug(f"📋 미들노트 기준 필터링: {len(filtered_perfumes)}개")

            if brand_filters:
                filtered_perfumes = [p for p in filtered_perfumes if p["brand"] in brand_filters]
                logger.debug(f"📋 브랜드 필터링 후: {len(filtered_perfumes)}개")

            if not filtered_perfumes:
                logger.error("❌ 필터링 결과 없음")
                raise HTTPException(status_code=404, detail="조건에 맞는 향수를 찾을 수 없습니다")

            # 4. GPT 프롬프트 생성
            products_text = "\n".join([
                f"{p['id']}. {p['name_kr']} ({p['brand']}): {p.get('main_accord', '향 정보 없음')}"
                for p in filtered_perfumes[:30]  # 최대 10개로 제한
            ])

            template = self.prompt_loader.get_prompt("recommendation")
            names_prompt = (
                f"{template['description']}\n"
                f"{template['rules']}\n"
                f"사용자의 원본 입력: {user_input}\n\n"
                f"아래 입력을 한국어로 번역한 후 향수를 추천하세요:\n"
                f"{user_input}\n\n"
                f"추출된 키워드: {products_text}\n"
                f"향수의 브랜드 이름은 포함하지 않은 이름만 최대 3개 추천해주세요.\n\n"
                f"- content: 추천 이유와 사용 상황, 향수들의 공통적인 느낌을 함께 적어주세요.\n\n"
                f"- line_id: 추천된 향수들의 공통적인 계열 아이디를 작성합니다.\n"
                "아래 JSON 형식으로만 응답하세요:\n"
                "```json\n"
                "{\n"
                '  "recommendations": [\n'
                '    {\n'
                '      "name": "블랑쉬 오 드 퍼퓸",\n'
                '      "reason": "깨끗한 머스크와 은은한 백합이 어우러져, 갓 세탁한 새하얀 리넨처럼 부드럽고 신선한 느낌을 선사합니다. 피부에 밀착되는 듯한 가벼운 향이 오래 지속되며, 자연스럽고 단정한 분위기를 연출합니다.",\n'
                '      "situation": "아침 샤워 후 상쾌한 기분을 유지하고 싶을 때, 오피스에서 단정하면서도 은은한 존재감을 남기고 싶을 때"\n'
                '    },\n'
                '    {\n'
                '      "name": "실버 마운틴 워터 오 드 퍼퓸",\n'
                '      "reason": "상큼한 시트러스와 신선한 그린 티 노트가 조화를 이루며, 알프스의 깨끗한 샘물을 연상시키는 맑고 청량한 느낌을 줍니다. 우디한 베이스가 잔잔하게 남아 차분한 매력을 더합니다.",\n'
                '      "situation": "운동 후 땀을 씻어내고 개운한 느낌을 유지하고 싶을 때, 더운 여름날 시원하고 깨끗한 인상을 주고 싶을 때"\n'
                '    },\n'
                '    {\n'
                '      "name": "재즈 클럽 오 드 뚜왈렛",\n'
                '      "reason": "달콤한 럼과 부드러운 바닐라가 타바코의 스모키함과 어우러져, 클래식한 재즈 바에서 오래된 가죽 소파에 앉아 칵테일을 마시는 듯한 분위기를 연출합니다. 깊고 따뜻한 향이 감각적인 무드를 더합니다.",\n'
                '      "situation": "여유로운 저녁 시간, 칵테일 바나 조용한 라운지에서 세련된 분위기를 연출하고 싶을 때, 가을과 겨울철 따뜻하고 매혹적인 향을 원할 때"\n'
                '    }\n'
                '  ]\n'
                '}\n'
                'content: "깨끗한 리넨의 산뜻함, 신선한 자연의 청량감, 그리고 부드러운 따뜻함이 조화롭게 어우러진 세련되고 감각적인 향입니다."\n'
                'line_id: 14\n'
                "```"
            )

            try:
                logger.info("🔄 향수 추천 처리 시작")
                
                # 1. GPT 응답 받기
                logger.info("🤖 GPT 응답 요청")
                response_text = self.gpt_client.generate_response(names_prompt)
                logger.debug(f"📝 GPT 원본 응답:\n{response_text}")

                # 2. JSON 파싱
                try:
                    # 마크다운 코드 블록 제거
                    if '```' in response_text:
                        parts = response_text.split('```')
                        for part in parts:
                            if '{' in part and '}' in part:
                                response_text = part.strip()
                                if response_text.startswith('json'):
                                    response_text = response_text[4:].strip()
                                break

                    # JSON 구조 추출
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if (start_idx == -1 or end_idx <= start_idx):
                        raise ValueError("JSON 구조를 찾을 수 없습니다")
                        
                    json_str = response_text[start_idx:end_idx]
                    logger.debug(f"📋 추출된 JSON:\n{json_str}")
                    
                    gpt_response = json.loads(json_str)
                    logger.info("✅ JSON 파싱 성공")

                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON 파싱 오류: {e}")
                    logger.error(f"📄 파싱 시도한 텍스트:\n{json_str if 'json_str' in locals() else 'None'}")
                    raise ValueError("JSON 파싱 실패")

                # 3. 추천 목록 생성
                recommendations = []
                for rec in gpt_response.get("recommendations", []):
                    matched_perfume = next(
                        (p for p in filtered_perfumes if p["name_kr"] == rec["name"]), 
                        None
                    )

                    if matched_perfume:
                        recommendations.append({
                            "id": matched_perfume["id"],
                            "name": matched_perfume["name_kr"], 
                            "brand": matched_perfume["brand"],
                            "reason": rec.get("reason", "추천 이유 없음"),
                            "situation": rec.get("situation", "사용 상황 없음")
                        })

                if not recommendations:
                    logger.error("❌ 유효한 추천 결과 없음")
                    raise ValueError("유효한 추천 결과가 없습니다")

                # 4. 공통 line_id 찾기
                common_line_id = self.get_common_line_id(recommendations)
                logger.info(f"✅ 공통 계열 ID: {common_line_id}")

                return {
                    "recommendations": recommendations,
                    "content": gpt_response.get("content", "추천 분석 실패"),
                    "line_id": common_line_id
                }

            except ValueError as ve:
                logger.error(f"❌ 추천 처리 오류: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                logger.error(f"❌ 예상치 못한 오류: {e}")
                raise HTTPException(status_code=500, detail="추천 생성 실패")

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            raise HTTPException(status_code=500, detail="추천 JSON 파싱 실패")
        except Exception as e:
            logger.error(f"추천 생성 오류: {str(e)}")
            raise HTTPException(status_code=500, detail="추천 생성 실패")    