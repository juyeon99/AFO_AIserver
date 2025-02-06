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
            logger.info(f"Detected intent: {intent}")  # 의도 감지 결과

            if "1" in intent:
                return "recommendation", None

            return "chat", None

        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            raise HTTPException(status_code=500, detail="Failed to classify user intent.")

    def extract_keywords_from_input(self, user_input: str) -> str:
        """사용자 입력에서 키워드를 추출하는 함수"""
        try:
            # 1. 프롬프트 생성
            logger.info("🔍 키워드 추출 시작")
            keywords_prompt = (
                "다음은 향수 추천 요청입니다. 이 요청에서 주요 키워드를 추출하세요. "
                "키워드는 시트러스, 플로럴, 우디, 머스크, 스파이시, 구르망과 같은 향 노트나 샤넬, 디올과 같은 브랜드일 수 있습니다.\n\n"
                "또한 남자다운, 여성스러운 등과 같은 요청은 적합한 향 노트로 변환하여 반환하세요.\n\n"
                f"요청: {user_input}\n\n"
                "추출된 키워드를 JSON 형식으로 반환하세요. 반드시 아래 형식을 따르세요:\n"
                "```json\n"
                "{ \"keywords\": \"시트러스 / 아쿠아틱 / 그린\" }\n"
                "```\n"
            )

            # 2. GPT 호출
            logger.info("🤖 GPT 키워드 추출 요청")
            response_text = self.gpt_client.generate_response(keywords_prompt).strip()
            logger.debug(f"📝 GPT 원본 응답:\n{response_text}")

            # 3. JSON 추출 및 파싱
            try:
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()

                parsed_response = json.loads(response_text)
                keywords_str = parsed_response.get('keywords', '').strip()

                if not keywords_str:
                    raise ValueError("🚨 'keywords' 필드를 찾을 수 없습니다.")

                logger.info(f"✅ 추출된 키워드: {keywords_str}")
                return keywords_str

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON 파싱 오류: {e}")
                logger.error(f"📄 파싱 시도한 텍스트: {response_text}")
                raise ValueError(f"🚨 JSON 파싱 실패: {e}")

        except Exception as e:
            logger.error(f"❌ 키워드 추출 오류: {e}")
            raise ValueError(f"🚨 키워드 추출 실패: {str(e)}")

    def generate_recommendation_response(self, user_input: str) -> dict:
        """사용자 요청 기반 향수 추천"""
        try:
            logger.info(f"Processing recommendation for user input: {user_input}")

            # 1. Load cached perfumes
            all_perfumes = self.db_service.load_cached_perfume_data()
            if not all_perfumes:
                raise HTTPException(status_code=404, detail="추천 가능한 향수를 찾을 수 없습니다")

            # 2. Extract keywords
            keywords = self.extract_keywords_from_input(user_input)
            logger.info(f"Extracted keywords: {keywords}")

            # 3. Filter perfumes (keywords 기반 필터링)
            filtered_perfumes = [
                p for p in all_perfumes
                if any(keyword.lower() in p.get('main_accord', '').lower() or 
                    keyword.lower() in p.get('brand', '').lower() for keyword in keywords.split(" / "))
            ]
            if not filtered_perfumes:
                raise HTTPException(status_code=404, detail="사용자 요청에 맞는 향수를 찾을 수 없습니다")

            # 4. GPT 프롬프트 생성
            products_text = "\n".join([
                f"{p['id']}. {p['name_kr']} ({p['brand']}): {p.get('main_accord', '향 정보 없음')}"
                for p in filtered_perfumes[:150]  # 최대 3개 추천
            ])

            template = self.prompt_loader.get_prompt("recommendation")
            names_prompt = (
                f"{template['description']}\n"
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
                'content: "추천 이유와 사용 상황과 향수들의 공통적인 느낌 함께 적어주세요."\n'
                'line_id: 14\n'
                "```"
            )

            # 5. Get GPT response and parse JSON
            response_text = self.gpt_client.generate_response(names_prompt)
            logger.debug(f"Raw GPT response: {response_text}")

            try:
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()

                gpt_response = json.loads(response_text)

                # 6. 추천 향수 ID 매칭
                recommendations = []
                for rec in gpt_response.get("recommendations", []):
                    matched_perfume = next((p for p in filtered_perfumes if p['name_kr'] == rec["name"]), None)
                    
                    if matched_perfume:
                        recommendations.append({
                            "id": matched_perfume["id"],
                            "name": matched_perfume["name_kr"],
                            "brand": matched_perfume["brand"],
                            "reason": rec.get("reason", "추천 이유 없음"),
                            "situation": rec.get("situation", "사용 상황 없음")
                        })

                # 7. 만약 추천된 향수가 데이터에 없으면 빈 값 반환
                if not recommendations:
                    raise ValueError("추천된 향수가 데이터에서 찾을 수 없습니다.")

                # 추천 결과에서 공통 계열 ID 찾기
                line_id = self.get_common_line_id(recommendations)
                
                return {
                    'recommendations': recommendations,
                    'content': gpt_response.get('content', '추천 분석 실패'),
                    'line_id': line_id  # 동적으로 계산된 line_id 사용
                }

            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {e}")
                logger.error(f"정제된 응답: {response_text}")
                raise ValueError("JSON 파싱 실패")

        except ValueError as ve:
            logger.error(f"추천 생성 오류: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
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

            # 1. 향수 목록 텍스트 생성
            perfume_names = [f"{rec['name']} ({rec['id']})" for rec in recommendations]
            prompt = (
                f"다음 향수 목록을 보고 공통된 계열 ID를 예측해주세요.\n"
                f"향수 목록: {', '.join(perfume_names)}\n"
                """가능한 계열 ID: 
                1. spicy : 후추나 시나몬 같은 향신료에서 느껴지는 따뜻하고 자극적인 향. 강렬하면서도 대담한 느낌으로, 어두운 조명 아래 와인 한 잔을 즐기는 우아한 저녁 모임에 어울립니다. 따뜻한 니트나 가죽 ...
                2. fruitry : 잘 익은 과일의 달콤함과 상큼함이 조화를 이루는 향. 햇살이 비치는 공원에서 피크닉을 즐기거나, 친구들과 가벼운 브런치를 즐길 때 어울립니다. 발랄하고 사랑스러운 이미지를 연출하는 데...
                3. Citrus : 갓 짜낸 오렌지 주스처럼 톡 쏘는 상쾌함과 신선함을 주는 향. 무더운 여름날 시원한 레모네이드를 마시는 순간이나, 활기찬 아침 출근길에 어울립니다. 에너지를 북돋아 주는 활력의 계열입...
                4. Green : 비 온 뒤 숲속의 맑은 공기나, 잘 정돈된 잔디밭 위의 신선함을 담은 향. 여유로운 산책이나 차분한 독서 시간과 잘 어울리며, 자연 속에 있는 듯한 평온함을 줍니다.
                5. Aldehyde : 고급스러운 비누향과 깨끗함이 돋보이는 향. 깔끔한 화이트 셔츠나 포멀한 정장 차림에 잘 어울리며, 모던하고 세련된 이미지를 강조합니다. 회의실에서의 자신감 넘치는 프레젠테이션이나, ...
                6. Aquatic : 시원한 바닷바람과 맑은 물방울을 연상시키는 청량한 향. 해변에서 여름 바캉스를 즐기거나, 수영장에서 느껴지는 잔잔한 물결의 고요함을 담았습니다. 더운 날의 피로를 씻어내는 상쾌함을 ...
                7. Fougere : 부드럽고 편안한 느낌을 주는 라벤더와 오크모스의 조화로운 향. 클래식한 이탈리안 정원에서 느껴지는 고급스러운 감성과 여유로움을 떠올리게 합니다. 남성적인 매력을 부드럽게 표현할 때...
                8. Gourmand : 따뜻한 초콜릿 케이크나 갓 구운 쿠키처럼 달콤하고 유혹적인 향. 디저트 카페에서 느낄 수 있는 포근한 향기로, 사랑스럽고 다정한 이미지를 연출합니다. 겨울 저녁에 특히 잘 어울리는 계열...
                9. Woody : 나무의 따뜻함과 자연스러운 우아함이 느껴지는 향. 모닥불 옆에서 느껴지는 평온함과, 빈티지 가구에서 풍기는 고급스러운 분위기를 연상시킵니다. 클래식한 스타일을 선호하는 이들에게 추...
                10. Oriental : 달콤하고 부드러운 바닐라와 고혹적인 앰버가 어우러져 센슈얼한 매력을 발산하는 향. 붉은 실크 드레스나 황혼의 무드에 어울리며, 이국적이고 매혹적인 분위기를 만들어냅니다.
                11. Floral : 화사한 꽃다발처럼 우아하고 여성스러운 향. 봄날 벚꽃 아래를 걷는 듯한 로맨틱한 분위기를 만들어주며, 부드러운 드레스 차림에 잘 어울립니다.
                12. Musk : 따뜻하면서도 부드럽게 이성을 자극하는 매혹적인 향. 깨끗하고 포근한 머스크의 베이스는, 자연스러운 아름다움을 강조하는 분위기를 연출합니다. 특별한 데이트나 은밀한 저녁 시간에 적합...
                13. Powdery : 포근한 담요처럼 감싸주는 부드럽고 따뜻한 향. 차분하고 편안한 분위기를 연출하며, 휴식이 필요한 공간이나 밤 시간에 잘 어울립니다.
                14. Tobacco Leathe : 스모키한 담배와 묵직한 가죽 향이 어우러져 강렬한 남성미를 느낄 수 있는 향. 한겨울 가죽 재킷을 입고 있는 도시 남성의 세련된 매력을 연상시킵니다. 중성적이면서도 고급스러운 분위기를... 중 하나를 선택하세요.\n"
                f"JSON 형식으로만 응답하세요:\n"
                "{\n"
                '  "line_id": 3\n'
                "}"""
            )

            # 2. GPT 요청
            logger.info("🤖 GPT 요청 시작")
            response = self.gpt_client.generate_response(prompt)
            logger.debug(f"📝 GPT 응답:\n{response}")

            # 3. JSON 파싱
            try:
                # 3-1. JSON 구조 추출
                clean_response = response.strip()
                if '```' in clean_response:
                    parts = clean_response.split('```')
                    for part in parts:
                        if '{' in part and '}' in part:
                            clean_response = part.strip()
                            if clean_response.startswith('json'):
                                clean_response = clean_response[4:].strip()
                            break

                # 3-2. JSON 경계 찾기
                start_idx = clean_response.find('{')
                end_idx = clean_response.rfind('}') + 1
                if start_idx == -1 or end_idx <= start_idx:
                    raise ValueError("JSON 구조를 찾을 수 없습니다")

                # 3-3. JSON 파싱
                json_str = clean_response[start_idx:end_idx]
                logger.debug(f"📋 추출된 JSON:\n{json_str}")
                
                response_data = json.loads(json_str)
                line_id = int(response_data.get("line_id", 1))
                
                logger.info(f"✅ GPT가 예측한 공통 계열 ID: {line_id}")
                return line_id

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ JSON 파싱 오류: {e}")
                logger.error(f"📄 파싱 시도한 응답:\n{response}")
                return 1

        except Exception as e:
            logger.error(f"❌ 계열 ID 검색 오류: {e}")
            return 1

