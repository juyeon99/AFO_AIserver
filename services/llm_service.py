import json, random
import logging, chromadb, json
from typing import Optional, Tuple
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from fastapi import HTTPException
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

chroma_client = chromadb.PersistentClient(path="chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS")

class LLMService:
    def __init__(self, gpt_client: GPTClient, db_service: DBService, prompt_loader: PromptLoader):
        self.gpt_client = gpt_client
        self.db_service = db_service
        self.prompt_loader = prompt_loader

        self.all_diffusers = self.db_service.load_cached_diffuser_data()
        self.diffuser_scent_descriptions = self.db_service.load_diffuser_scent_cache()

        if not self.all_diffusers:
            raise RuntimeError("No diffuser data available for initialization.")

        # Initialize vector database
        self.collection = self.initialize_vector_db(self.all_diffusers, self.diffuser_scent_descriptions)

    def process_input(self, user_input: Optional[str] = None, image_caption: Optional[str] = None) -> Tuple[str, Optional[int]]:
        """
        사용자 입력을 분석하여 의도를 분류합니다.
        """
        try:
            logger.info(f"Received user input: {user_input}")  # 입력 로그

            # 의도 분류 프롬프트
            intent_prompt = (
                f"user_input: {user_input}\n"
                f"image_caption: {image_caption}\n"
                f"다음 사용자의 의도를 분류하세요.\n\n"
                f"일반적인 키워드라고 볼 수 없는 향수 추천은 (2) 일반 대화로 분류해야 합니다.\n\n"
                f"예시) user_input = 나 오늘 기분이 너무 우울해. 그래서 이런 기분을 떨쳐낼 수 있는 플로럴 계열의 향수를 추천해줘 (1) 향수 추천 \n"
                f"예시) user_input = 향수를 추천받고 싶은데 뭐 좋은 거 있어? (2) 일반 대화\n"
                f"예시) user_input = 향수를 추천해주세요. 라면 (2) 일반 대화로 분류해야 합니다.\n\n"
                f"의도: (1) 향수 추천, (2) 일반 대화, (3) 패션 향수 추천, (4) 인테리어 기반 디퓨저 추천, (5) 테라피 목적 향수/디퓨저 추천"
            )

            intent = self.gpt_client.generate_response(intent_prompt).strip()
            logger.info(f"Detected intent: {intent}")  # 의도 감지 결과

            if "1" in intent:
                logger.info("💡 일반 향수 추천 실행")
                return "recommendation", self.generate_recommendation_response(user_input, image_caption)

            if "3" in intent:
                logger.info("👕 패션 기반 향수 추천 실행 (mode는 recommendation 유지)")
                return "recommendation", self.fashion_based_generate_recommendation_response(user_input, image_caption)
            
            if "4" in intent:
                logger.info("🏡 공간 기반 디퓨저 추천 실행")
                return "recommendation", self.generate_interior_design_based_recommendation_response(user_input, image_caption)
            
            if "5" in intent:
                logger.info("🌏 테라피 목적 향수 추천 실행")
                return "recommendation", self.generate_therapeutic_purpose_recommendation_response(user_input, image_caption)

            return "chat", self.generate_chat_response(user_input)

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to classify user intent.")

    def extract_keywords_from_input(self, user_input: Optional[str] = None, image_caption: Optional[str] = None, language: Optional[str] = None) -> dict:
        """사용자 입력에서 계열과 브랜드를 분석하고 계열 ID와 브랜드 리스트를 반환하는 함수"""
        try:
            if user_input is not None:
                logger.info(f"🔍 입력된 user_input에서 향 계열과 브랜드 분석 시작: {user_input}")
            elif image_caption is not None:
                logger.info(f"🔍 image_caption: {image_caption}")
            elif language is not None:
                logger.info(f"🔍 language: {language}")

            # 1. DB에서 계열 및 브랜드 데이터 가져오기
            line_data = self.db_service.fetch_line_data()
            line_mapping = {line["name"]: line["id"] for line in line_data}
            
            brand_list = self.db_service.fetch_kr_brands()
            
            # if language == "korean":
            #     brand_list = self.db_service.fetch_kr_brands()
            # else:
            #     brand_list = self.db_service.load_brand_en_list()
            
            # 2. GPT를 이용해 입력에서 향 계열과 브랜드 추출
            keywords_prompt = (
                "The following is a perfume recommendation request. Extract the fragrance family and brand names from the user_input and image_caption.\n"
                f"### Fragrance families(line): {', '.join(line_mapping.keys())}\n\n"
                f"### Brand list: {', '.join(brand_list)}\n\n"

                "### Additional rules:\n"
                "- If the user_input and the image_caption is a description of a fashion style, use the corresponding fragrance family from the following fashion styles.\n"
                "- If the user_input is a description of a date or a specific situation, use the corresponding fragrance family for the situation.\n"
                "- Infer the user's style or vibe from the user_input or image_caption (e.g., sporty, romantic, vintage, etc.) and recommend a fragrance family(line) based on that.\n"
                "- If the user specifies a brand, include it only if it exists in the Brand list. If the mentioned brand is not in the Brand list, do not include it in the output.\n"
                "- Exclude any brands that the user explicitly does not want.\n\n"

                "### Fashion style to output fragrance family(line) mapping example:\n"
                "1. Fashion style: Casual style -> line: **Fruity**\n"
                "2. Fashion style: Dandy Casual -> line: **Woody**\n"
                "3. Fashion style: American Casual -> line: **Green**\n"
                "4. Fashion style: Classic -> line: **Woody**\n"
                "5. Fashion style: Business Formal -> line: **Musk**\n"
                "6. Fashion style: Business Casual -> line: **Citrus**\n"
                "7. Fashion style: Gentle Style -> line: **Powdery**\n"
                "8. Fashion style: Street -> line: **Spicy**\n"
                "9. Fashion style: Techwear -> line: **Aromatic**\n"
                "10. Fashion style: Gorp Core -> line: **Green**\n"
                "11. Fashion style: Punk Style -> line: **Tobacco Leather**\n"
                "12. Fashion style: Sporty -> line: **Citrus**\n"
                "13. Fashion style: Runner Style -> line: **Aquatic**\n"
                "14. Fashion style: Tennis Look -> line: **Fougere**\n"
                "15. Fashion style: Vintage -> line: **Oriental**\n"
                "16. Fashion style: Romantic Style -> line: **Floral**\n"
                "17. Fashion style: Bohemian -> line: **Musk**\n"
                "18. Fashion style: Retro Fashion -> line: **Aldehyde**\n"
                "19. Fashion style: Modern -> line: **Woody**\n"
                "20. Fashion style: Minimal -> line: **Powdery**\n"
                "21. Fashion style: All Black Look -> line: **Tobacco Leather**\n"
                "22. Fashion style: White Tone Style -> line: **Musk**\n"
                "23. Fashion style: Avant-garde -> line: **Tobacco Leather**\n"
                "24. Fashion style: Gothic Style -> line: **Oriental**\n"
                "25. Fashion style: Cosplay -> line: **Gourmand**\n\n"

                "### Few-shot examples:\n")
            
            if language == "korean":
                keywords_prompt += (
                    "#### Example 1:\n"
                    "user_input: '비즈니스 미팅에 어울리는 향수가 뭐가 있나요? 주로 샤넬 제품을 선호합니다.'\n"
                    "Expected Output:\n"
                    "{\n"
                    '  "line": "Musk",\n'
                    '  "brands": ["샤넬"]\n'
                    "}\n\n"

                    "#### Example 2:\n"
                    "user_input: '아침 조깅할 때 사용할 시원하고 깨끗한 향을 찾고 있어요.'\n"
                    "Expected Output:\n"
                    "{\n"
                    '  "line": "Aquatic",\n'
                    '  "brands": []\n'
                    "}\n\n"

                    "#### Example 3:\n"
                    "user_input: '빈티지한 패션을 즐겨 입어요. 고풍스럽고 우아한 향수를 추천해 주세요.'\n"
                    "Expected Output:\n"
                    "{\n"
                    '  "line": "Oriental",\n'
                    '  "brands": []\n'
                    "}\n\n"

                    "#### Example 4:\n"
                    "user_input: '로맨틱한 분위기의 데이트에 어울리는 향수를 추천해 주세요. 조말론과 딥디크 제품을 좋아해요.'\n"
                    "Expected Output:\n"
                    "{\n"
                    '  "line": "Floral",\n'
                    '  "brands": ["조 말론", "딥티크"]\n'
                    "}\n\n"

                    "#### Example 5:\n"
                    "user_input: '나는 디올 향수는 별로 안 좋아해. 포멀한 수트와 어울리는 여성스러운 향을 추천해 줘.'\n"
                )
            else:
                keywords_prompt += (
                    "#### Example 1:\n"  
                    "user_input: 'What are some perfumes suitable for a business meeting? I usually prefer Chanel products.'\n"  
                    "Expected Output:\n"  
                    "{\n"  
                    '  "line": "Musk",\n'  
                    '  "brands": ["샤넬"]\n'  
                    "}\n\n"  

                    "#### Example 2:\n"  
                    "user_input: 'I'm looking for a fresh and clean scent to use during my morning jog.'\n"  
                    "Expected Output:\n"  
                    "{\n"  
                    '  "line": "Aquatic",\n'  
                    '  "brands": []\n'  
                    "}\n\n"  

                    "#### Example 3:\n"  
                    "user_input: 'I enjoy wearing vintage fashion. Please recommend a sophisticated and elegant perfume.'\n"  
                    "Expected Output:\n"  
                    "{\n"  
                    '  "line": "Oriental",\n'  
                    '  "brands": []\n'  
                    "}\n\n"  

                    "#### Example 4:\n"  
                    "user_input: 'Please recommend a perfume suitable for a romantic date. I like Jo Malone and Diptyque products.'\n"  
                    "Expected Output:\n"  
                    "{\n"  
                    '  "line": "Floral",\n'  
                    '  "brands": ["조 말론", "딥티크"]\n'  
                    "}\n\n"  

                    "#### Example 5:\n"  
                    "user_input: 'I don't really like Dior perfumes. Please recommend a feminine scent that goes well with a formal suit.'\n"
                )
            
            keywords_prompt += (
                "Expected Output:\n"
                "{\n"
                '  "line": "Musk",\n'
                '  "brands": []\n'
                "}\n\n"

                "### Important rule: The 'line' must **never** be null. It should always correspond to **one of Fragrance families(line)**.\n"
                "### NOTE: The 'brands' list contains the brands the user wants. It can be empty if the user does not specify any brand. Exclude any brands that the user explicitly does not want. If a brand is mentioned but is not in the Brand list, do not include it in the output. If a brand is included, it must exactly match the name as listed in the Brand list.\n\n"
            )

            if user_input is not None:
                keywords_prompt += f"### user_input: {user_input}\n\n"
            
            if image_caption is not None:
                keywords_prompt += f"### image_caption: {image_caption}\n\n"

            keywords_prompt += (   
                "### The output format must be **JSON**:\n"
                "{\n"
                '  "line": "Woody",\n'
                '  "brands": []\n'
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

    def generate_recommendation_response(self, user_input: Optional[str] = None, image_caption: Optional[str] = None, language: Optional[str] = None) -> dict:
        """middle note를 포함한 향수 추천"""
        try:
            if user_input is not None:
                logger.info(f"🔄 추천 처리 시작 - user_input: {user_input}")
            if image_caption is not None:
                logger.info(f"🔄 image_caption: {image_caption}")
            if language is not None:
                logger.info(f"🔄 language: {language}")
            
            # 1. 키워드 추출
            logger.info("🔍 키워드 추출 시작")
            extracted_data = self.extract_keywords_from_input(user_input=user_input, image_caption=image_caption, language=language)
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

            # 프롬프트 생성
            template = self.prompt_loader.get_prompt("recommendation")
            names_prompt = (
                f"{template['description']}\n"
                f"{template['rules']}"
            )

            if user_input is not None:
                names_prompt += f"\n### user_input: {user_input}\n"
            
            if image_caption is not None:
                names_prompt += f"\n### image_caption: {image_caption}\n"

            # 3. 향수 필터링
            logger.info("🔍 향수 필터링 시작")
            filtered_perfumes = self.db_service.get_perfumes_by_middle_notes(spice_ids)
            logger.debug(f"📋 미들노트 기준 필터링: {len(filtered_perfumes)}개")

            if brand_filters:
                brand_filtered_perfumes = [p for p in filtered_perfumes if p["brand"] in brand_filters]
                logger.debug(f"📋 브랜드 필터링 후: {len(brand_filtered_perfumes)}개")

                if len(brand_filtered_perfumes) < 3:
                    logger.debug("📋 브랜드 필터링 결과가 3개 미만이므로 브랜드 필터링을 하지 않은 미들노트 기준 결과를 사용합니다.")
                    random.shuffle(filtered_perfumes)
                    filtered_perfumes = filtered_perfumes[:25]

                    names_prompt += f"\n### Preferred brand: {brand_filters}\n"
                    names_prompt += (
                        "- If a brand in 'Preferred brand' matches a brand from the database, recommend perfumes from that brand.\n"
                        "- If no matching brand is found, recommend based on user_input and image_caption(if exists) without considering the brand.\n\n"
                    )

                    for perfume in brand_filtered_perfumes:
                        if perfume not in filtered_perfumes:
                            filtered_perfumes.append(perfume)   # 브랜드 필터링을 하지 않은 미들노트 기준 결과에 brand_filtered_perfumes의 제품이 포함되지 않은 경우 포함
                else:
                    random.shuffle(brand_filtered_perfumes)
                    filtered_perfumes = brand_filtered_perfumes[:25]

            if not filtered_perfumes:
                logger.error("❌ 필터링 결과 없음")
                raise HTTPException(status_code=404, detail="조건에 맞는 향수를 찾을 수 없습니다.")

            # 4. GPT 프롬프트 생성
            products_text = "\n".join([
                f"{p['id']}. {p['name_kr'] if language == 'korean' else p['name_en']} ({p['brand']}): {p.get('main_accord', 'No scent information available')}"
                for p in filtered_perfumes
            ])

            names_prompt += (
                f"### Products list (id. name (brand): main_accord): \n{products_text}\n\n"
                f"Recommend up to 3 fragrance names that do not include brand names.\n\n"
                f"- content: Please include the reason for the recommendation, the situation it suits, and the common feel of the perfumes in {language.upper()}.\n\n"

                f"### Important Rule: You must respond only **in {language.upper()}**\n\n"

                "Respond only in the following JSON format:\n"
                "```json\n"
                "{\n"
                '  "recommendations": [\n'
                '    {\n'
            )
            
            if language == "korean":
                names_prompt += (
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
                    '  ],\n'
                    '  "content": "깨끗한 리넨의 산뜻함, 신선한 자연의 청량감, 그리고 부드러운 따뜻함이 조화롭게 어우러진 세련되고 감각적인 향입니다."'
                )
            
            else:
                names_prompt += (
                    '      "name": "BLANCHE EDP",\n'  
                    '      "reason": "A blend of clean musk and delicate lily creates a soft and fresh sensation, reminiscent of freshly laundered white linen. The light scent adheres closely to the skin and lasts for a long time, giving off a natural and neat impression.",\n'  
                    '      "situation": "When you want to maintain a refreshing feeling after a morning shower, or when you want to leave a subtle yet polished presence in the office."\n'  
                    '    },\n'  
                    '    {\n'  
                    '      "name": "SILVER MOUNTAIN WATER EDP",\n'  
                    '      "reason": "A harmony of zesty citrus and fresh green tea notes evokes the image of pristine alpine spring water, delivering a clear and refreshing sensation. A woody base lingers subtly, adding a calm and composed charm.",\n'  
                    '      "situation": "When you want to feel refreshed after a workout, or when you want to give off a cool and clean impression on a hot summer day."\n'  
                    '    },\n'  
                    '    {\n'  
                    '      "name": "REPLICA JAZZ CLUB EDT",\n'  
                    '      "reason": "Sweet rum and smooth vanilla blend with the smoky depth of tobacco, creating an atmosphere reminiscent of lounging in an old leather armchair at a classic jazz bar while sipping a cocktail. The deep and warm scent enhances a sensual mood.",\n'  
                    '      "situation": "During a relaxed evening, when you want to create a sophisticated vibe at a cocktail bar or a quiet lounge, or when you desire a warm and captivating fragrance in the fall and winter."\n'  
                    '    }\n'  
                    '  ],\n'  
                    '  "content": "A sophisticated and sensual fragrance that harmoniously blends the crisp freshness of clean linen, the invigorating clarity of nature, and a gentle warmth."'
                )
            
            names_prompt += (
                '}\n'
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
                    name_key = "name_kr" if language == "korean" else "name_en"

                    matched_perfume = next(
                        (p for p in filtered_perfumes if p[name_key] == rec["name"]),
                        None
                    )

                    if matched_perfume:
                        recommendations.append({
                            "id": matched_perfume["id"],
                            "name": matched_perfume[name_key], 
                            "brand": matched_perfume["brand"],
                            "reason": rec.get("reason", "-"),
                            "situation": rec.get("situation", "-")
                        })

                if not recommendations:
                    logger.error("❌ 유효한 추천 결과 없음")
                    raise ValueError("유효한 추천 결과가 없습니다")

                # 4. 공통 line_id 찾기
                common_line_id = self.get_common_line_id(recommendations)
                logger.info(f"✅ 공통 계열 ID: {common_line_id}")

                return {
                    "recommendations": recommendations,
                    "content": gpt_response.get("content", "-"),
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
        """추천된 product들의 공통 계열 ID를 찾는 함수"""
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
                    
                # product 계열 정보 생성
                line_info = "\n".join([
                    f"{line['id']}: {line['name']} - {line.get('content', '설명 없음')}"
                    for line in line_data
                ])

                # 2. product 목록 생성
                product_list = "\n".join([
                    f"{rec['id']}. {rec['name']}: {rec['reason']}" 
                    for rec in recommendations
                ])
                logger.debug(f"📋 분석할 product 목록: {product_list}")

                # 3. GPT 프롬프트 생성 
                prompt = (
                    f"다음 향수/디퓨저 목록을 보고 가장 적합한 계열 ID를 선택해주세요.\n\n"
                    f"향수/디퓨저 목록:\n{product_list}\n\n"
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
        
    def fashion_based_generate_recommendation_response(self, user_input: Optional[str] = None, image_caption: Optional[str] = None, language: Optional[str] = None) -> dict:
        """middle note를 포함한 향수 추천"""
        try:
            logger.info(f"🔄 추천 처리 시작 - 입력: {user_input}")

            # 1. 키워드 추출 
            logger.info("🔍 키워드 추출 시작")
            extracted_data = self.extract_keywords_from_input(user_input, image_caption, language)
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

            # 프롬프트 생성
            template = self.prompt_loader.get_prompt("recommendation")
            names_prompt = (
                f"{template['description']}\n"
                f"{template['rules']}\n\n"
            )

            if user_input is not None:
                names_prompt += f"### user_input: {user_input}\n"
            if image_caption is not None:
                names_prompt += f"### image_caption: {image_caption}\n"

            # 3. 향수 필터링
            logger.info("🔍 향수 필터링 시작")
            filtered_perfumes = self.db_service.get_perfumes_by_middle_notes(spice_ids)
            logger.debug(f"📋 미들노트 기준 필터링: {len(filtered_perfumes)}개")

            if brand_filters:
                brand_filtered_perfumes = [p for p in filtered_perfumes if p["brand"] in brand_filters]
                logger.debug(f"📋 브랜드 필터링 후: {len(brand_filtered_perfumes)}개")

                if len(brand_filtered_perfumes) < 3:
                    logger.debug("📋 브랜드 필터링 결과가 3개 미만이므로 브랜드 필터링을 하지 않은 미들노트 기준 결과를 사용합니다.")
                    random.shuffle(filtered_perfumes)
                    filtered_perfumes = filtered_perfumes[:25]

                    names_prompt += f"\n### Preferred brand: {brand_filters}\n"
                    names_prompt += (
                        "- If a brand in 'Preferred brand' matches a brand from the database, recommend perfumes from that brand.\n"
                        "- If no matching brand is found, recommend based on user_input and image_caption(if exists) without considering the brand.\n\n"
                    )

                    for perfume in brand_filtered_perfumes:
                        if perfume not in filtered_perfumes:
                            filtered_perfumes.append(perfume)   # 브랜드 필터링을 하지 않은 미들노트 기준 결과에 brand_filtered_perfumes의 제품이 포함되지 않은 경우 포함
                else:
                    random.shuffle(brand_filtered_perfumes)
                    filtered_perfumes = brand_filtered_perfumes[:25]

            if not filtered_perfumes:
                logger.error("❌ 필터링 결과 없음")
                raise HTTPException(status_code=404, detail="조건에 맞는 향수를 찾을 수 없습니다.")

            # 4. GPT 프롬프트 생성
            products_text = "\n".join([
                f"{p['id']}. {p['name_kr']} ({p['brand']}): {p.get('main_accord', '향 정보 없음')}"
                for p in filtered_perfumes
            ])

            names_prompt += (
                f"### Products list (id. name (brand): main_accord): \n{products_text}\n\n"
                f"Recommend up to 3 perfume names without including the brand names.\n\n"
                f"Note: The recommendations should refer to the user_input, image_caption, and extracted keywords. The image_caption describes the person's outfit, and the recommended perfumes should match the described outfit.\n"
                f"- content: Please include the reason for the recommendation, the situation it suits, and the common feel of the perfumes in {language.upper()}.\n\n"
                f"### Important Rule: You must respond only **in {language.upper()}**\n\n"
                "Respond only in the following JSON format:\n"
                "```json\n"
                "{\n"
                '  "recommendations": [\n'
                '    {\n'
            )
            
            if language == "korean":
                names_prompt += (
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
                    '  ],\n'
                    '  "content": "깨끗한 리넨의 산뜻함, 신선한 자연의 청량감, 그리고 부드러운 따뜻함이 조화롭게 어우러진 세련되고 감각적인 향입니다."'
                )
            
            else:
                names_prompt += (
                    '      "name": "BLANCHE EDP",\n'  
                    '      "reason": "A blend of clean musk and delicate lily creates a soft and fresh sensation, reminiscent of freshly laundered white linen. The light scent adheres closely to the skin and lasts for a long time, giving off a natural and neat impression.",\n'  
                    '      "situation": "When you want to maintain a refreshing feeling after a morning shower, or when you want to leave a subtle yet polished presence in the office."\n'  
                    '    },\n'  
                    '    {\n'  
                    '      "name": "SILVER MOUNTAIN WATER EDP",\n'  
                    '      "reason": "A harmony of zesty citrus and fresh green tea notes evokes the image of pristine alpine spring water, delivering a clear and refreshing sensation. A woody base lingers subtly, adding a calm and composed charm.",\n'  
                    '      "situation": "When you want to feel refreshed after a workout, or when you want to give off a cool and clean impression on a hot summer day."\n'  
                    '    },\n'  
                    '    {\n'  
                    '      "name": "REPLICA JAZZ CLUB EDT",\n'  
                    '      "reason": "Sweet rum and smooth vanilla blend with the smoky depth of tobacco, creating an atmosphere reminiscent of lounging in an old leather armchair at a classic jazz bar while sipping a cocktail. The deep and warm scent enhances a sensual mood.",\n'  
                    '      "situation": "During a relaxed evening, when you want to create a sophisticated vibe at a cocktail bar or a quiet lounge, or when you desire a warm and captivating fragrance in the fall and winter."\n'  
                    '    }\n'  
                    '  ],\n'  
                    '  "content": "A sophisticated and sensual fragrance that harmoniously blends the crisp freshness of clean linen, the invigorating clarity of nature, and a gentle warmth."'
                )
            
            names_prompt += (
                '}\n'
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
                    name_key = "name_kr" if language == "korean" else "name_en"

                    matched_perfume = next(
                        (p for p in filtered_perfumes if p[name_key] == rec["name"]),
                        None
                    )

                    if matched_perfume:
                        recommendations.append({
                            "id": matched_perfume["id"],
                            "name": matched_perfume[name_key], 
                            "brand": matched_perfume["brand"],
                            "reason": rec.get("reason", "-"),
                            "situation": rec.get("situation", "-")
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

    def initialize_vector_db(self, diffuser_data, diffuser_scent_descriptions):
        """Initialize Chroma DB and store embeddings."""
        logger.info(f"Initializing Chroma DB.")
        collection = chroma_client.get_or_create_collection(name="embeddings", embedding_function=embedding_function)

        # Fetch existing IDs from the collection
        existing_ids = set()
        try:
            results = collection.get()
            existing_ids.update(results["ids"])
        except Exception as e:
            logger.error(f"Error fetching existing IDs: {e}")

        # Insert vectors for each diffuser if not already in collection
        for diffuser in diffuser_data:
            if str(diffuser["id"]) in existing_ids:
                # logger.info(f"Skipping diffuser ID {diffuser['id']} (already in collection).")
                continue
            
            logger.info(f"Inserting vectors for ID {diffuser['id']}.")
            scent_description = diffuser_scent_descriptions.get(diffuser["id"], "")

            combined_text = f"{diffuser['brand']}\n{diffuser['name_kr']} ({diffuser['name_en']})\n{scent_description}"

            # Store in Chroma
            collection.add(
                documents=[combined_text],
                metadatas=[{"id": diffuser["id"], "name_kr": diffuser["name_kr"], "name_en": diffuser["name_en"], "brand": diffuser["brand"], "category_id": diffuser["category_id"], "scent_description": scent_description}],
                ids=[str(diffuser["id"])]
            )
        logger.info(f"Diffuser data have been embedded and stored in Chroma.")

        return collection
    
    def get_distinct_brands(self, product_data):
        """Return all distinct diffuser brands from the product data."""
        # 디퓨저는 개수가 적으므로 디퓨저 데이터만 가지고 브랜드 추출할 수 있는 함수 따로 생성
        brands = set()
        for product in product_data:
            brands.add(product.get("brand", "Unknown"))
        return brands
    
    def get_fragrance_recommendation(self, user_input: Optional[str] = None, image_caption: Optional[str] = None, language: Optional[str] = None):
        # GPT에게 user input과 image caption 전달 후 어울리는 향에 대한 설명 반환(특정 브랜드 있으면 맨 앞에 적게끔 요청.)
        existing_brands = self.get_distinct_brands(self.all_diffusers)
        brands_str = ", ".join(existing_brands)

        fragrance_description_prompt = f"""You are a fragrance expert with in-depth knowledge of various scents. Based on the User Input and Image Caption, **imagine** and provide a fragrance scent description that matches the room's description and the user's request. Focus more on the User Input. Your task is to creatively describe a fragrance that would fit well with the mood and characteristics of the room as described in the caption, as well as the user's scent preference. Do not mention specific diffuser or perfume products.

            ### Instructions:
            - Existing Brands: {brands_str}
            1. **If a specific brand is mentioned**, check if it exists in the list of existing brands above. If it does, acknowledge the brand name without referring to any specific product and describe a fitting scent that aligns with the user's request.  
            **IF THE BRAND IS MENTIONED IN THE USER INPUT BUT IS NOT FOUND IN THE EXISTING BRANDS LIST, START BY 'Not Found' TO SAY THE BRAND DOES NOT EXIST.**
            2. **If the brand is misspelled or doesn't exist**, please:
                - Correct the spelling if the brand is close to an existing brand (e.g., "아쿠아 파르마" -> "아쿠아 디 파르마").
                - **IF THE BRAND IS MENTIONED IN THE USER INPUT BUT IS NOT FOUND IN THE EXISTING BRANDS LIST, START BY 'Not Found' TO SAY THE BRAND DOES NOT EXIST.** Then, recommend a suitable fragrance based on the context and preferences described in the user input.
            3. Provide the fragrance description in **{language.upper()}**, focusing on key scent notes and creative details that align with the mood and characteristics described in the user input and image caption. Do **not mention specific diffuser or perfume products.**

            ### Example Responses:

            #### Example 1 (when a brand is mentioned, but with a minor spelling error):
            - User Input: {"아쿠아 파르마의 우디한 베이스를 가진 디퓨저를 추천해줘." if language == "korean" else "Recommend a diffuser with a woody base from Acqua di Parma."}
            - Image Caption: The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa.
            - Response:
            Brand: 아쿠아 디 파르마
            Scent Description: {"우디한 베이스에 따뜻하고 자연스러운 분위기를 더하는 향이 어울립니다. 은은한 샌들우드와 부드러운 시더우드가 조화를 이루며, 가벼운 머스크와 드라이한 베티버가 깊이를 더합니다. 가벼운 허브와 상쾌한 시트러스 노트가 은은하게 균형을 이루며 여유롭고 세련된 분위기를 연출합니다." if language == "korean" else "A fragrance that enhances a woody base with a warm and natural atmosphere is a perfect match. Subtle sandalwood and soft cedarwood create harmony, while light musk and dry vetiver add depth. Delicate herbs and refreshing citrus notes subtly balance the composition, evoking a relaxed and sophisticated ambiance."}

            #### Example 2 (when no brand is mentioned):
            - User Input: {"우디한 베이스를 가진 디퓨저를 추천해줘." if language == "korean" else "Recommend a diffuser with a woody base."}
            - Image Caption: The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa.
            - Response:
            Brand: None
            Scent Description: {"우디한 베이스에 따뜻하고 자연스러운 분위기를 더하는 향이 어울립니다. 은은한 샌들우드와 부드러운 시더우드가 조화를 이루며, 가벼운 머스크와 드라이한 베티버가 깊이를 더합니다. 가벼운 허브와 상쾌한 시트러스 노트가 은은하게 균형을 이루며 여유롭고 세련된 분위기를 연출합니다." if language == "korean" else "A fragrance that enhances a woody base with a warm and natural atmosphere is a perfect match. Subtle sandalwood and soft cedarwood create harmony, while light musk and dry vetiver add depth. Delicate herbs and refreshing citrus notes subtly balance the composition, evoking a relaxed and sophisticated ambiance."}

            #### Example 3 (when a brand is mentioned but not in the list of existing brands):
            - User Input: {"샤넬 브랜드 제품의 우디한 베이스를 가진 디퓨저를 추천해줘." if language == "korean" else "Recommend a diffuser with a woody base from Chanel."}
            - Image Caption: The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa.
            - Response:
            Brand: Not Found
            Scent Description: {"우디한 베이스에 따뜻하고 자연스러운 분위기를 더하는 향이 어울립니다. 은은한 샌들우드와 부드러운 시더우드가 조화를 이루며, 가벼운 머스크와 드라이한 베티버가 깊이를 더합니다. 가벼운 허브와 상쾌한 시트러스 노트가 은은하게 균형을 이루며 여유롭고 세련된 분위기를 연출합니다." if language == "korean" else "A fragrance that enhances a woody base with a warm and natural atmosphere is a perfect match. Subtle sandalwood and soft cedarwood create harmony, while light musk and dry vetiver add depth. Delicate herbs and refreshing citrus notes subtly balance the composition, evoking a relaxed and sophisticated ambiance."}
            """
        
        if user_input is not None:
            fragrance_description_prompt += f"\n### User Input: {user_input}"
        if image_caption is not None:
            fragrance_description_prompt += f"\n### Image Caption: {image_caption}"
        fragrance_description_prompt += f"\n### Response: "
        
        fragrance_description = self.gpt_client.generate_response(fragrance_description_prompt).strip()
        return fragrance_description
    
    def generate_interior_design_based_recommendation_response(self, user_input: Optional[str] = None, image_caption: Optional[str] = None, language: Optional[str] = None) -> dict:
        """공간 사진 기반 디퓨저 추천"""
        try:
            logger.info(f"🏠 공간 사진 기반 디퓨저 추천 시작: {user_input}")
            fragrance_description = self.get_fragrance_recommendation(user_input=user_input, image_caption=image_caption, language=language)

            try:
                diffusers_result = self.collection.query(
                    query_texts=[fragrance_description],
                    n_results=10,
                    # where={"brand": "딥티크"},
                    # where_document={"$contains":"프루티"}
                )

                # Extracting data from the query result
                ids = diffusers_result["ids"][0]
                documents = diffusers_result["documents"][0]
                metadata = diffusers_result["metadatas"][0]

                name_key = "name_kr" if language == "korean" else "name_en"

                for i in range(len(ids)):
                    product_id = ids[i]
                    name = metadata[i][name_key]
                    brand = metadata[i]["brand"]
                    scent_description = metadata[i]["scent_description"]
                    logger.info(f"Query Result - id: {product_id}. {name} ({brand})\n{scent_description}\n")

                diffusers_text = "\n".join([
                    f"{metadata[i]['id']}. {metadata[i][name_key]} ({metadata[i]['brand']}): {metadata[i]['scent_description']}"
                    for i in range(len(metadata))
                ])
            except Exception as e:
                logger.error(f"Error during Chroma query: {e}")
                diffusers_result = None

            template = self.prompt_loader.get_prompt("diffuser_recommendation")
            diffuser_prompt = (
                f"{template['description']}\n"
                f"{template['rules']}\n"
            )

            if user_input is not None:
                diffuser_prompt += f"### user_input: {user_input}\n"
            if image_caption is not None:
                diffuser_prompt += f"### image_caption: {image_caption}\n"

            diffuser_prompt += (
                f"Diffusers List (id. name (brand): scent_description):\n{diffusers_text}\n"
                f"Recommend up to 2 diffusers, including only the id and name, excluding the brand name.\n\n"
                f"Note: The recommendations should refer to the user_input, image_caption(if exists). The image_caption describes the interior design or a space, and the recommended diffusers should match the described interior design.\n"
                f"- content: Based on the user_input and image_caption, please include the reason for the recommendation, the situation it suits, and the common feel of the diffusers in {language.upper()}.\n\n"
                f"### Important Rule: You must respond only **in {language.upper()}**\n\n"

                "Respond only in the following JSON format:\n"
                "```json\n"
                "{\n"
                '  "recommendations": [\n'
                '    {\n'
            )

            if language == "korean":
                diffuser_prompt += (
                    '      "id": 1503,\n'
                    '      "name": "블랑쉬 오 드 퍼퓸",\n'
                    '      "reason": "깨끗한 머스크와 은은한 백합이 어우러져, 갓 세탁한 새하얀 리넨처럼 부드럽고 신선한 느낌을 선사. 피부에 밀착되는 듯한 가벼운 향이 오래 지속되며, 자연스럽고 단정한 분위기를 연출함.",\n'
                    '      "situation": "아침 샤워 후 상쾌한 기분을 유지하고 싶을 때, 오피스에서 단정하면서도 은은한 존재감을 남기고 싶을 때"\n'
                    '    },\n'
                    '    {\n'
                    '      "id": 1478,\n'
                    '      "name": "실버 마운틴 워터 오 드 퍼퓸",\n'
                    '      "reason": "상큼한 시트러스와 신선한 그린 티 노트가 조화를 이루며, 알프스의 깨끗한 샘물을 연상시키는 맑고 청량한 느낌을 줌. 우디한 베이스가 잔잔하게 남아 차분한 매력을 더함.",\n'
                    '      "situation": "운동 후 땀을 씻어내고 개운한 느낌을 유지하고 싶을 때, 더운 여름날 시원하고 깨끗한 인상을 주고 싶을 때"\n'
                    '    },\n'
                    '  ],\n'
                    '  "content": "깨끗한 리넨의 산뜻함, 신선한 자연의 청량감, 그리고 부드러운 따뜻함이 조화롭게 어우러진 세련되고 감각적인 향입니다."'
                )
            
            else:
                diffuser_prompt += (
                    '      "name": "BLANCHE EDP",\n'  
                    '      "reason": "A blend of clean musk and delicate lily creates a soft and fresh sensation, reminiscent of freshly laundered white linen. The light scent adheres closely to the skin and lasts for a long time, giving off a natural and neat impression.",\n'  
                    '      "situation": "When you want to maintain a refreshing feeling after a morning shower, or when you want to leave a subtle yet polished presence in the office."\n'  
                    '    },\n'  
                    '    {\n'  
                    '      "name": "SILVER MOUNTAIN WATER EDP",\n'  
                    '      "reason": "A harmony of zesty citrus and fresh green tea notes evokes the image of pristine alpine spring water, delivering a clear and refreshing sensation. A woody base lingers subtly, adding a calm and composed charm.",\n'  
                    '      "situation": "When you want to feel refreshed after a workout, or when you want to give off a cool and clean impression on a hot summer day."\n'  
                    '    },\n' 
                    '  ],\n'  
                    '  "content": "A sophisticated and sensual fragrance that harmoniously blends the crisp freshness of clean linen, the invigorating clarity of nature, and a gentle warmth."'
                )
            
            diffuser_prompt += (
                '}\n'
                "```"
            )

            try:
                logger.info("🔄 디퓨저 추천 처리 시작")
                
                # 1. GPT 응답 받기
                logger.info("🤖 GPT 응답 요청")
                response_text = self.gpt_client.generate_response(diffuser_prompt)
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
                    name_key = "name_kr" if language == "korean" else "name_en"

                    matched_diffuser = next(
                        (d for d in self.all_diffusers if d["id"] == rec["id"]), 
                        None
                    )

                    if matched_diffuser:
                        recommendations.append({
                            "id": matched_diffuser["id"],
                            "name": matched_diffuser[name_key], 
                            "brand": matched_diffuser["brand"],
                            "reason": rec.get("reason", "-"),
                            "situation": rec.get("situation", "-")
                        })

                if not recommendations:
                    logger.error("❌ 유효한 추천 결과 없음")
                    raise ValueError("유효한 추천 결과가 없습니다")

                # 4. 공통 line_id 찾기
                common_line_id = self.get_common_line_id(recommendations)
                logger.info(f"✅ 공통 계열 ID: {common_line_id}")

                response_data = {
                    "recommendations": recommendations,
                    "content": gpt_response.get("content", "추천 분석 실패"),
                    "line_id": common_line_id
                }

                return response_data

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

    def decide_product_category(self, user_input: str, language: str) -> int:
        """
        This function uses GPT to determine whether the user is asking for a diffuser (2) or a perfume (1).
        It returns 2 (default) if the user asks for neither or if there is an error.
        """
        product_category_prompt = f"""
            Given the user input, determine whether the user is asking for a diffuser or a perfume recommendation. 
            1. {"향수 추천" if language == "korean" else "Perfume"}
            2. {"디퓨저 추천" if language == "korean" else "Diffuser"}

            If the user is asking for a diffuser recommendation, return 2.
            If the user is asking for a perfume recommendation, return 1.
            If the request is for neither, return 2 by default.

            Respond with only a number: 1 or 2.

            ### Example 1:
            User input: {"기분 좋은 향기가 나는 디퓨저를 추천해줘." if language == "korean" else "Recommend a diffuser with a pleasant fragrance."}
            Response: 2

            ### Example 2:
            User input: {"피로를 풀어주는 향수를 추천해줘." if language == "korean" else "Recommend a perfume that helps relieve fatigue."}
            Response: 1

            ### Example 3:
            User input: {"스트레스 해소에 도움이 되는 제품을 추천해줘." if language == "korean" else "Recommend a product that helps reduce stress."}
            Response: 2

            ### Important Rule:
            If the user input mentions {"향수" if language == "korean" else "perfume"}, return 1.
            If the input mentions {"디퓨저" if language == "korean" else "diffuser"} or does not mention either, return 2.

            User input: {user_input}
            Response: 
        """

        category_id = 2  # Default category_id is set to 2 (for diffuser)
        product_category_response = self.gpt_client.generate_response(product_category_prompt).strip()

        if product_category_response:
            try:
                category_id = int(product_category_response)
            except ValueError:
                category_id = 2
        logger.info(f"🎀 카테고리 id: {category_id}")

        return category_id

    def analyze_user_input_effect(self, user_input: str, language: str) -> list:
        """
        This function uses GPT to analyze the user's input and return a list of primary effects (as integers).
        It returns [3] (Refreshing) by default in case of an error or invalid response.
        """
        user_input_effect_prompt = f"""
        Given the user input "{user_input}", identify the primary effect or effects the user is seeking among the following categories:
        1. Stress Reduction ()
        2. Happiness (행복)
        3. Refreshing (리프레시)
        4. Sleep Aid (수면)
        5. Concentration (집중)
        6. Energy Boost (에너지)
        Respond with only a number or numbers separated by commas.

        ### Example 1:
        Input: {"요즘 잠을 잘 못자는데 수면에 도움이 될만한 디퓨저를 추천해줘." if language == "korean" else "I haven't been sleeping well lately. Recommend a diffuser that can help with sleep."}
        Output: 4

        ### Example 2:
        Input: {"기분전환에 도움이 될만한 향수를 추천해줘." if language == "korean" else "Recommend a perfume that can help lift my mood."}
        Output: 3, 6

        ### Example 3:
        Input: {"요즘 스트레스를 받았더니 좀 기분이 쳐져. 기분을 업되게 할만한 향수를 추천해줘." if language == "korean" else "I've been feeling down due to stress. Recommend a perfume that can boost my mood."}
        Output: 1"""

        user_input_effect_response = self.gpt_client.generate_response(user_input_effect_prompt).strip()
        try:
            user_input_effect_list = [int(x) for x in user_input_effect_response.split(',')]
        except ValueError:
            user_input_effect_list = [3]  # Default to [3] (Refreshing) if there's an error
        logger.info(f"🎀 사용자 요구 효능 리스트: {user_input_effect_list}")

        return user_input_effect_list

    def generate_therapeutic_purpose_recommendation_response(self, user_input: Optional[str] = None, image_caption: Optional[str] = None, language: Optional[str] = None) -> dict:
        """테라피 기반 향수/디퓨저 추천"""
        try:
            if user_input is not None:
                logger.info(f"🌏 테라피 기반 향수/디퓨저 추천 user_input: {user_input}")
            if image_caption is not None:
                logger.info(f"🌏 image_caption: {image_caption}")
            if language is not None:
                logger.info(f"🌏 language: {language}")
            
            category_id = 2
            user_input_effect_list = [3]

            if user_input is not None:
                # Get the product category
                category_id = self.decide_product_category(user_input, language)

                # Analyze user input effects
                user_input_effect_list = self.analyze_user_input_effect(user_input, language)

            if category_id == 2:
                all_products = self.all_diffusers
                template = self.prompt_loader.get_prompt("diffuser_recommendation")
            else:
                all_products = self.db_service.load_cached_perfume_data()
                template = self.prompt_loader.get_prompt("recommendation")
                
            # Load note cache and spice therapeutic effect cache
            note_cache = self.db_service.load_cached_note_data()
            spice_effect_cache = self.db_service.load_cached_spice_therapeutic_effect_data()
            
            # Create a map of spice_id to effect
            spice_effect_map = {entry["id"]: entry["effect"] for entry in spice_effect_cache}
            
            # Filter notes that have note_type as "MIDDLE" or "SINGLE" and match user input effects
            valid_notes = [note for note in note_cache 
                        if note["note_type"] in ("MIDDLE", "SINGLE") 
                        and spice_effect_map.get(note["spice_id"]) in user_input_effect_list]
            
            # Get product IDs that match the valid notes
            valid_product_ids = {note["product_id"] for note in valid_notes}
            
            # Filter all_products based on valid product IDs
            filtered_products = [product for product in all_products if product["id"] in valid_product_ids]
            random.shuffle(filtered_products)
            selected_products = filtered_products[:20]
            
            purposes = {
                1: "Stress Reduction",
                2: "Happiness",
                3: "Refreshing",
                4: "Sleep Aid",
                5: "Concentration",
                6: "Energy Boost"
            }

            purpose = ", ".join([purposes[i] for i in user_input_effect_list])
            logger.info(f"🦢 테라피 효능: {purpose}")

            # Create a map of spice_id to name for easy lookup
            spice_name_map = {entry["id"]: entry["name_en"] for entry in spice_effect_cache}

            # Create a mapping of product_id to its MIDDLE/SINGLE spices
            product_spice_map = {}

            for note in note_cache:
                if note["note_type"] in ("MIDDLE", "SINGLE") and note["product_id"] in {p["id"] for p in selected_products}:
                    product_id = note["product_id"]
                    spice_name = spice_name_map.get(note["spice_id"], "Unknown Spice")

                    if product_id not in product_spice_map:
                        product_spice_map[product_id] = []
                    
                    product_spice_map[product_id].append(spice_name)

            name_key = "name_kr" if language == "korean" else "name_en"

            products_text = "\n".join(
                f"{product['id']}. {product[name_key]} ({product['brand']}): {', '.join(product_spice_map.get(product['id'], []))}"
                for product in selected_products
            )

            prompt = (
                f"{template['description']}\n"
                f"{template['rules']}\n"
            )

            if user_input is not None:
                prompt += f"### user_input: {user_input}\n"
            if image_caption is not None:
                prompt += f"### image_caption: {image_caption}\n"

            if category_id == 2:
                prompt += (
                    f"Diffuser Recommendation Purpose: {purpose}\n\n"
                    f"Based on the user_input, image_caption(if exists) and recommendation purpose, recommend diffusers:\n"
                    f"Diffuser list (id. name (brand): spices):\n{products_text}\n"
                    f"Recommend 2 diffusers, including only the id and name, excluding the brand name.\n\n"
                    f"- content: Based on the user_input, image_caption and recommendation purpose, provide reasons for the recommendation, usage scenarios, and the common impression of the diffusers according to the recommendation purpose.\n"
                    "The following example shows a diffuser recommendation for stress relief as the recommendation purpose.\n"
                    f"### Important Rule: You must respond only **in {language.upper()}**\n\n"
                    "Respond only in the following JSON format:\n"
                    "```json\n"
                    "{\n"
                    '  "recommendations": [\n'
                    '    {\n'
                    '      "id": 1503,\n'
                )

                if language == "korean":
                    prompt += (
                        '      "name": "레지오 디퓨저",\n'
                        '      "reason": "라벤더와 베르가못의 조화로운 향이 마음을 편안하게 만들어 주며, 스트레스를 완화하는 데 도움을 줍니다. 은은한 자스민이 부드러운 플로럴 감각을 더해주고, 머스크의 포근한 잔향이 안정감을 선사하여 긴장된 몸과 마음을 편안하게 감싸줍니다. 하루의 피로를 풀고 휴식을 취하기에 적합한 향으로, 조용한 공간에서 마음을 진정시키고 싶을 때 추천합니다.",\n'
                        '      "situation": "하루 일과를 마친 후 편안한 휴식을 취하고 싶을 때, 조용한 공간에서 명상이나 독서를 하며 마음을 안정시키고 싶을 때, 또는 스트레스로 인해 쉽게 잠들기 어려운 밤에 숙면을 돕기 위해 사용"\n'
                        '    },\n'
                        '    {\n'
                        '      "id": 1518,\n'
                        '      "name": "카페 소사이어트 퍼퓸 건",\n'
                        '      "reason": "파촐리와 앰버의 따뜻하고 깊은 향이 몸과 마음을 편안하게 감싸주며, 라벤더의 부드럽고 허브 같은 향이 긴장을 완화하고 안정감을 줍니다. 은은하면서도 차분한 분위기를 연출하여 스트레스와 피로를 덜어주고 편안한 휴식을 돕습니다.",\n'
                        '      "situation": "하루를 마무리하며 조용한 휴식을 취하고 싶을 때, 따뜻한 조명 아래에서 독서를 하거나 명상을 할 때 사용하면 마음이 차분해지고 안정감을 느낄 수 있습니다. 또한 스트레스로 지친 날, 편안한 분위기 속에서 나만의 시간을 즐기고 싶을 때 활용하면 좋습니다."\n'
                        '    }\n'
                        '  ],\n'
                        '  "content": "부드럽고 따뜻한 향이 조화를 이루어 스트레스로 지친 마음을 편안하게 감싸줍니다. 포근하고 안정적인 잔향이 공간을 감싸며 긴장을 풀어주고, 차분하고 아늑한 분위기를 연출하여 하루의 피로를 덜어주는 데 도움을 주는 향입니다."'
                    )
                
                else:
                    prompt += (
                        '      "name": "DIFFUSER REGGIO",\n'  
                        '      "reason": "The harmonious blend of lavender and bergamot creates a calming atmosphere and helps relieve stress. The subtle touch of jasmine enhances the soft floral sensation, while the warm musk base provides a sense of stability, gently enveloping the body and mind in relaxation. This scent is perfect for unwinding after a long day, ideal for creating a tranquil space to soothe the mind.",\n'  
                        '      "situation": "When you want to relax after a long day, when you seek peace while meditating or reading in a quiet space, or when you have trouble falling asleep due to stress and need a fragrance to promote restful sleep."\n'  
                        '    },\n'  
                        '    {\n'  
                        '      "id": 1518,\n'  
                        '      "name": "CAFE SOCIETY PERFUME GUN",\n'  
                        '      "reason": "The warm and deep notes of patchouli and amber gently embrace the body and mind, while the soft, herbal scent of lavender helps relieve tension and instill a sense of calm. This delicate yet composed fragrance creates a soothing atmosphere, alleviating stress and fatigue to promote relaxation.",\n'  
                        '      "situation": "Ideal for winding down after a long day, reading or meditating under warm lighting to foster a sense of tranquility and stability. It is also perfect for those moments when you need to recharge in a cozy atmosphere after a stressful day."\n'  
                        '    }\n'  
                        '  ],\n'  
                        '  "content": "A harmonious blend of soft and warm scents that gently soothes a stressed mind. The comforting and stable base notes fill the space with a relaxing aura, easing tension and creating a calm, cozy ambiance to help relieve the fatigue of the day."'
                    )
                
            else:
                prompt += (
                    f"Perfume Recommendation Purpose: {purpose}\n\n"
                    f"Based on the user_input, image_caption(if exists) and recommendation purpose, recommend perfumes:\n"
                    f"Perfume list (id. name (brand): spices):\n{products_text}\n"
                    f"Recommend 3 perfumes, including only the id and name, excluding the brand name.\n\n"
                    f"- content: Based on the user_input, image_caption and recommendation purpose, provide reasons for the recommendation, usage scenarios, and the common impression of the perfumes according to the recommendation purpose.\n"
                    "The following example shows a perfume recommendation for stress relief as the recommendation purpose.\n"
                    f"### Important Rule: You must respond only **in {language.upper()}**\n\n"
                    "Respond only in the following JSON format:\n"
                    "```json\n"
                    "{\n"
                    '  "recommendations": [\n'
                    '    {\n'
                    '      "id": 403,\n'
                )

                if language == "korean":
                    prompt += (
                        '      "name": "쟈도르 롤러 펄 오 드 퍼퓸",\n'
                        '      "reason": "부드럽고 우아한 플로럴 향이 감각적으로 퍼지며, 긴장된 마음을 편안하게 진정시키는 데 도움을 줍니다. 풍성하고 따뜻한 꽃향기가 포근한 감성을 자아내어 스트레스 속에서도 안정감을 느낄 수 있도록 돕습니다. 자연스럽고 조화로운 향의 흐름이 마음을 부드럽게 어루만져 하루의 피로를 풀고 평온한 분위기를 연출합니다.",\n'
                        '      "situation": "하루를 마무리하며 편안한 시간을 보내고 싶을 때 사용하면 좋습니다. 저녁 휴식 시간에 가볍게 발라 깊은 숨을 들이마시면, 부드러운 꽃향기가 마음을 차분하게 감싸주며 스트레스를 완화하는 데 도움을 줍니다. 또한 따뜻한 차 한 잔과 함께 조용한 시간을 보내거나, 스파나 목욕 후 몸과 마음을 안정시키고 싶을 때 사용하면 더욱 편안한 분위기를 느낄 수 있습니다."\n'
                        '    },\n'
                        '    {\n'
                        '      "id": 765,\n'
                        '      "name": "위스퍼 인 라이브러리 오 드 뚜왈렛",\n'
                        '      "reason": "따뜻하고 부드러운 바닐라 향이 감각을 편안하게 감싸주며, 우디 노트와 시더우드의 차분한 깊이가 안정감을 더해줍니다. 은은한 페퍼의 가벼운 스파이시함이 부담스럽지 않게 조화를 이루어, 따뜻하면서도 고요한 분위기를 연출합니다. 이 향은 복잡한 생각을 정리하고 마음의 긴장을 풀어주는 데 도움을 주며, 조용한 순간을 더욱 편안하고 아늑하게 만들어줍니다.",\n'
                        '      "situation": "고요한 분위기 속에서 마음을 차분하게 가라앉히고 싶을 때 사용하기 좋습니다. 독서하며 깊은 몰입감을 느끼고 싶을 때, 비 오는 날 창가에서 따뜻한 차 한 잔과 함께 여유로운 시간을 보내고 싶을 때, 혹은 하루를 마무리하며 조용한 음악과 함께 긴장을 풀고 편안한 휴식을 취하고 싶을 때 어울리는 향입니다."\n'
                        '    }\n'
                        '    {\n'
                        '      "id": 694,\n'
                        '      "name": "베르가못 22 오 드 퍼퓸",\n'
                        '      "reason": "베르가못과 자몽의 상쾌하고 신선한 향이 기분을 전환시키고, 오렌지 블로섬과 페티그레인에서 느껴지는 부드러운 꽃향기가 마음을 편안하게 만들어 줍니다. 또한, 머스크와 앰버가 조화를 이루어 따뜻하고 안정적인 분위기를 조성하며, 시더와 베티버의 깊이 있는 향이 마음을 차분하게 안정시켜 스트레스를 해소하는 데 도움을 줍니다. 이 향수는 복잡한 생각을 정리하고 평온한 상태로 이끌어주는 효과가 있습니다.",\n'
                        '      "situation": "업무나 중요한 일로 인한 스트레스를 해소하고 싶을 때 혹은 긴장을 풀고 싶을 때 사용하면 좋습니다. 또한, 차 한 잔과 함께 여유로운 시간을 보내고 싶을 때나, 편안한 휴식을 취하고 싶을 때 이 향수를 뿌리면, 상쾌하면서도 안정감 있는 향이 마음을 진정시키고 편안한 분위기를 만들어 줍니다."\n'
                        '    }\n'
                        '  ],\n'
                        '  "content": "부드럽고 따뜻한 향들이 감각을 감싸며, 고요하고 차분한 분위기를 만들어 마음을 안정시킵니다. 향들이 자연스럽게 퍼지며 긴장을 풀어주고, 편안하고 평온한 시간을 만들어 줍니다. 이 디퓨저들은 복잡한 생각을 정리하고 마음을 진정시키는 데 도움을 주며, 하루의 스트레스를 해소할 수 있는 최적의 선택이 될 것입니다."'
                    )
                
                else:
                    prompt += (
                        '      "name": "J\'ADORE ROLLER PEARL EDP",\n'  
                        '      "reason": "The soft and elegant floral scent spreads sensually, helping to soothe a tense mind. The rich and warm floral fragrance evokes a sense of comfort, allowing you to feel at ease even in stressful moments. The natural and harmonious flow of scents gently embraces the mind, relieving the fatigue of the day and creating a serene atmosphere.",\n'  
                        '      "situation": "Perfect for unwinding at the end of the day. Applying it lightly during evening relaxation and taking a deep breath allows the soft floral scent to gently envelop the mind, helping to alleviate stress. It is also ideal for quiet moments with a warm cup of tea or after a spa or bath session to calm both body and mind and enhance a sense of tranquility."\n'  
                        '    },\n'  
                        '    {\n'  
                        '      "id": 765,\n'  
                        '      "name": "REPLICA WHISPERS IN LIBRARY Eau De Toilette",\n'  
                        '      "reason": "The warm and soft vanilla scent gently embraces the senses, while the deep and calming notes of wood and cedarwood enhance the feeling of stability. A subtle spiciness from pepper adds a balanced touch, creating a warm yet tranquil ambiance. This fragrance helps clear the mind of cluttered thoughts, eases tension, and makes quiet moments feel even more comforting and cozy.",\n'  
                        '      "situation": "Ideal for times when you want to calm your mind in a peaceful setting. It pairs well with deep immersion in reading, enjoying a warm cup of tea by the window on a rainy day, or unwinding with soft music at the end of the day for a relaxing and restful experience."\n'  
                        '    }\n'  
                        '    {\n'  
                        '      "id": 694,\n'  
                        '      "name": "BERGAMOTE 22 EDP",\n'  
                        '      "reason": "The refreshing and invigorating scent of bergamot and grapefruit uplifts the mood, while the soft floral notes of orange blossom and petitgrain bring a sense of calm. Musk and amber create a warm and stable atmosphere, complemented by the depth of cedar and vetiver, which help soothe the mind and relieve stress. This fragrance aids in organizing thoughts and fostering a sense of peace.",\n'  
                        '      "situation": "Great for relieving stress from work or important tasks, or simply for unwinding. It also enhances a relaxed mood when enjoyed with a cup of tea or during moments of restful solitude, as its fresh yet grounding scent calms the mind and sets a peaceful ambiance."\n'  
                        '    }\n'  
                        '  ],\n'  
                        '  "content": "Soft and warm scents gently embrace the senses, creating a serene and tranquil atmosphere. These fragrances naturally disperse, easing tension and fostering a sense of relaxation and peace. They are an excellent choice for clearing the mind, soothing the spirit, and relieving the stresses of the day."'
                    )
                
            prompt += (
                '}\n'
                "```"
            )
            
            try:
                logger.info("🔄 향수/디퓨저 추천 처리 시작")
                
                # 1. GPT 응답 받기
                logger.info("🤖 GPT 응답 요청")
                response_text = self.gpt_client.generate_response(prompt)
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
                    matched_product = next(
                        (d for d in selected_products if d["id"] == rec["id"]), 
                        None
                    )

                    if matched_product:
                        recommendations.append({
                            "id": matched_product["id"],
                            "name": matched_product[name_key], 
                            "brand": matched_product["brand"],
                            "reason": rec.get("reason", "-"),
                            "situation": rec.get("situation", "-")
                        })

                if not recommendations:
                    logger.error("❌ 유효한 추천 결과 없음")
                    raise ValueError("유효한 추천 결과가 없습니다")

                # 4. 공통 line_id 찾기
                common_line_id = self.get_common_line_id(recommendations)
                logger.info(f"✅ 공통 계열 ID: {common_line_id}")

                response_data = {
                    "recommendations": recommendations,
                    "content": gpt_response.get("content", "추천 분석 실패"),
                    "line_id": common_line_id
                }

                return response_data

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