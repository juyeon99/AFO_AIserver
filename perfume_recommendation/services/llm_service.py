import json
import logging
from typing import Optional, Tuple
from models.img_llm_client import GPTClient
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from fastapi import HTTPException
from collections import defaultdict
from difflib import SequenceMatcher

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

    def extract_keywords_from_input(self, user_input: str) -> list:
        """
        사용자 입력에서 주요 키워드를 GPT로 추출하고 정리합니다.
        """
        # GPT에게 보낼 프롬프트 생성
        keywords_prompt = (
            "다음은 향수 추천 요청입니다. 이 요청에서 주요 키워드를 추출하세요. "
            "키워드는 시트러스, 플로럴, 우디, 머스크, 스파이시, 구르망과 같은 향 노트나 샤넬, 디올과 같은 브랜드일 수 있습니다.\n\n"
            "또한 남자다운, 여성스러운 등과 같은 요청은 적합한 향 노트로 변환하여 반환하세요.\n\n"
            f"요청: {user_input}\n\n"
            "추출된 키워드를 JSON 형식으로 반환하세요:\n"
            """keywords": {
                "여성스러운": ["시트러스", "플로럴", "우디", "구르망", "머스크", "그린"],
                "남성스러운": ["머스크", "우디", "스파이시", "시트러스", "레더", "앰버"],
                "우아한": ["플로럴", "파우더리", "머스크", "화이트 플로럴"],
                "상쾌한": ["시트러스", "아쿠아틱", "그린", "프레시"],
                "달콤한": ["구르망", "프루티", "바닐라", "앰버"],
                "따뜻한": ["우디", "스파이시", "앰버", "레더"],
                "부드러운": ["파우더리", "화이트 플로럴", "머스크", "플로럴"],
                "강렬한": ["스파이시", "레더", "우디", "앰버", "타바코"],
                "캐주얼한": ["시트러스", "프루티", "아쿠아틱", "그린"],
                "고급스러운": ["앰버", "레더", "머스크", "우디", "플로럴"],
                "활동적인": ["아쿠아틱", "시트러스", "프레시 스파이시", "그린"]
            }
            "}"""
        )

        # GPT 호출
        logger.info("Sending keyword extraction request to GPT")
        response_text = self.gpt_client.generate_response(keywords_prompt).strip()
        if not response_text:
            raise ValueError("GPT 응답이 비어 있습니다")

        # JSON 파싱
        try:
            if '```' in response_text:
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text.split('\n', 1)[1]

            clean_response = response_text.strip()
            gpt_response = json.loads(clean_response)

            # 키워드 정리
            extracted_keywords_dict = gpt_response.get("keywords", {})
            if not extracted_keywords_dict:
                raise ValueError("GPT에서 키워드를 추출하지 못했습니다")

            # 키워드 값들을 하나의 리스트로 병합
            extracted_keywords = []
            for key, values in extracted_keywords_dict.items():
                extracted_keywords.extend(values)

            logger.info(f"추출된 키워드: {extracted_keywords}")
            return extracted_keywords

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.error(f"원본 응답: {response_text}")
            raise ValueError("GPT 응답을 JSON으로 파싱할 수 없습니다")

    
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

            # 3. Filter perfumes
            filtered_perfumes = [
                p for p in all_perfumes
                if any(keyword.lower() in p.get('main_accord', '').lower() or 
                    keyword.lower() in p.get('brand', '').lower() for keyword in keywords)
            ]
            if not filtered_perfumes:
                raise HTTPException(status_code=404, detail="사용자 요청에 맞는 향수를 찾을 수 없습니다")

            # 4. Generate GPT prompt
            products_text = "\n".join([
                f"{p['id']}. {p['name_kr']} ({p['brand']}): {p.get('main_accord', '향 정보 없음')}"
                for p in filtered_perfumes[:3]  # Limit to 3 perfumes
            ])

            template = self.prompt_loader.get_prompt("recommendation")
            names_prompt = (
                f"{template['description']}\n"
                f"사용자 요청: {user_input}\n"
                f"추출된 키워드: {products_text}\n"
                f"향수의 브랜드 이름은 들어가지 않은 이름만 최대 3개 추천해주세요. 각 향수는 사용자 요청에 따라 아래 정보를 포함합니다:\n"
                "- name: 추천된 향수의 이름\n"
                "- reason: 해당 향수를 추천하는 이유\n"
                "- situation: 해당 향수를 사용할 수 있는 상황\n\n"
                "아래 JSON 형식으로만 응답하세요:\n"
                "{\n"
                '  "recommendations": [\n'
                '    {\n'
                '      "name": "넘버5 오 드 퍼퓸",\n'
                '      "reason": "은(는) 강렬한 첫인상을 주며 포마드 헤어스타일과 깔끔한 정장에 어울리는 남성적인 향수입니다.",\n'
                '      "situation": "격식을 차린 행사나 특별한 날의 분위기에 잘 맞습니다."\n'
                '    }\n,'
                '    {\n'
                '      "name": "코코 마드모아젤 오 드 뚜왈렛",\n'
                '      "reason": "은(는) 강렬한 첫인상을 주며 포마드 헤어스타일과 깔끔한 정장에 어울리는 남성적인 향수입니다.",\n'
                '      "situation": "격식을 차린 행사나 특별한 날의 분위기에 잘 맞습니다."\n'
                '    }\n'
                '  ]\n'
                'content: "추천 결과에 대한 추가 설명이나 정보를 입력하세요."\n'
                "}"
            )

            # 5. Get GPT response and parse JSON
            response_text = self.gpt_client.generate_response(names_prompt)
            logger.debug(f"Raw GPT response: {response_text}")

            try:
                # Clean response and extract JSON
                clean_response = response_text.strip()
                if '```' in clean_response:
                    parts = clean_response.split('```')
                    for part in parts:
                        if '{' in part and '}' in part:
                            clean_response = part.strip()
                            break

                # Find JSON boundaries
                start_idx = clean_response.find('{')
                end_idx = clean_response.rfind('}') + 1
                if start_idx == -1 or end_idx <= start_idx:
                    raise ValueError("JSON 구조를 찾을 수 없습니다")

                json_str = clean_response[start_idx:end_idx]
                gpt_response = json.loads(json_str)

            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {e}")
                logger.error(f"정제된 응답: {clean_response}")
                raise ValueError("JSON 파싱 실패")

            # 6. Process recommendations
            recommendations = []
            for rec in gpt_response.get('recommendations', []):
                matched_perfumes = self.find_similar_perfumes(filtered_perfumes, rec.get('name', ''))
                for perfume in matched_perfumes:
                    recommendations.append({
                        'id': perfume['id'],
                        'name': perfume['name_kr'],
                        'brand': perfume['brand'],
                        'reason': rec.get('reason', '추천 이유 없음'),
                        'situation': rec.get('situation', '사용 상황 없음')
                    })

            if not recommendations:
                raise ValueError("유효한 추천 결과가 없습니다")

            # 7. Return final response
            return {
                'mode': 'recommendation',
                'recommendations': recommendations,
                'content': gpt_response.get('content', '추천 분석 실패')
            }

        except ValueError as ve:
            logger.error(f"추천 생성 오류: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"추천 생성 오류: {str(e)}")
            raise HTTPException(status_code=500, detail="추천 생성 실패")

    def find_similar_perfumes(self, perfumes, target_names: list[str], limit: int = 3) -> list[dict]:
        """향수 목록에서 지정된 이름과 유사한 향수를 찾아 반환"""
        
        def normalize_name(name: str) -> str:
            """향수 이름 정규화"""
            concentrations = ["퍼퓸", "오 드 퍼퓸", "오 드 뚜왈렛", "엥땅스", "로 프리베"]
            name = name.lower()
            for c in concentrations:
                name = name.replace(c.lower(), "")
            return name.strip()

        def get_priority(name: str) -> int:
            """부향률 우선순위"""
            if "퍼퓸" in name: return 1
            if "오 드 퍼퓸" in name: return 2
            if "오 드 뚜왈렛" in name: return 3
            return 4

        results = []
        seen_bases = set()
        
        # 각 타겟 이름에 대해 유사도 검사
        for target in target_names:
            normalized_target = normalize_name(target)
            
            # 유사도가 높은 향수 찾기
            matches = []
            for perfume in perfumes:
                name_kr = perfume.get('name_kr', '')
                normalized_name = normalize_name(name_kr)
                
                # 이미 처리된 기본 이름은 건너뛰기
                if normalized_name in seen_bases:
                    continue
                    
                similarity = SequenceMatcher(None, normalized_name, normalized_target).ratio()
                if similarity > 0.6:
                    matches.append((similarity, get_priority(name_kr), perfume))
            
            # 유사도와 부향률 우선순위로 정렬
            matches.sort(key=lambda x: (-x[0], x[1]))
            
            if matches:
                best_match = matches[0][2]
                results.append(best_match)
                seen_bases.add(normalize_name(best_match['name_kr']))
                
            # 결과가 limit개에 도달하면 중단
            if len(results) >= limit:
                break

        logger.debug(f"Selected perfumes: {[p['name_kr'] for p in results]}")
        return results[:limit]
