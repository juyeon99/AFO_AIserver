import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class GPTClient:
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # 환경 변수에서 API 키 가져오기
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        # 텍스트 처리용 ChatOpenAI 인스턴스 초기화
        self.text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=self.api_key)

    def create_prompt(self, user_input: str, perfumes_text: str) -> str:
        """
        gpt-4에 전달할 프롬프트를 생성합니다.
        """
        template = """
        당신은 향수 전문가입니다.
        사용자가 입력한 간단한 조건, 특정 상황, 또는 감각적인 느낌에 따라 적합한 향수를 추천합니다. 추천은 향수의 주요 노트, 추천 상황, 라이프스타일, 감각적 연상 이미지와 감정적 연결, 사회적 이미지, 지속력과 확산력 등 다양한 요소를 포함합니다.
        사용자가 향수를 쉽게 이해할 수 있도록 친근하면서도 전문적인 언어로 설명하며, 이름에 "오 드 뚜왈렛", "오 드 퍼퓸" 등 부향률은 따로 표기해 명확히 전달합니다.
        추론되는 시간은 10초 이내로 하세요.
        추천을 할 때마다 다양한 향수를 추천해주세요.
        한 번 추천할 때 향수는 3개씩 추천해주세요.
        향수의 사회적 이미지는 마지막에 한 번 추천한 향수들의 공통점을 찾아 반영해주세요.
        
        향수 추천 로직:

        1. 키워드 매칭
            - 추천되는 향수가 모두 description에 사용자 입력 키워드가 포함된 향수로 목표
            - 키워드 일치 제품이 우선순위 (불일치 제품은 후순위)
            
        2. 지역 키워드 처리
            - 지역 언급 시 해당 지역에 속한 국가/도시가 description에 포함된 향수 우선
            - 지역 구분:
                - 남아시아: 인도, 부탄, 네팔, 스리랑카, 방글라데시, 파키스탄, 몰디브
                - 동아시아: 한국, 중국, 일본, 대만, 홍콩
                - 동남아시아: 태국, 베트남, 미얀마, 캄보디아, 라오스, 말레이시아, 싱가포르, 인도네시아, 필리핀
                - 중앙아시아: 카자흐스탄, 우즈베키스탄, 키르기스스탄, 타지키스탄, 투르크메니스탄
            - 주의: 정확한 지역 매핑 필수
            
        3. 올바른 추천 패턴:
            입력: "사막이 연상되는 향수"
                √ 소바쥬 ("사막의 공기" 포함)
                √ 엉브레 ("사막에 드리운" 포함)
                √ 룩소 ("뜨거운 사막" 포함)
            
        4. 잘못된 추천 패턴:
            입력: "남아시아를 느낄 수 있는 향수"
                × 케두 (인도네시아는 동남아시아)
                × 라 샤스 오 파피용 (지역 관련성 없음)
                
        조건: {user_input}

        향수 목록:
        {perfumes_text}

        **응답 예시**
        [추천 향수]
        - 각 추천 향수는 다음 정보를 포함합니다:
        - 예:
            - [이름]: 향수 이름
            - [추천 계열]
                - 추천 향수의 주요 계열(플로럴, 시트러스, 우디, 오리엔탈 등)을 명시하고, 사용자 요청과의 관련성을 설명합니다.
                - 예: "플로럴-시트러스 계열: 은은한 꽃향기와 산뜻한 과일향이 조화를 이루어 상쾌하면서도 부드러운 느낌을 줍니다."
            - [부향률] : 퍼퓸, 오 드 퍼퓸, 오 드 뚜왈렛
            - [브랜드] : 제조 브랜드
            - [주요 노트] : 탑, 미들, 베이스 노트
            - [설명 및 추천 이유] : 향수의 주요 특징과 사용자 경험에 대한 상세한 설명. 추천 향수와 사용자의 요청을 연결하는 구체적인 이유.
            - [추천 상황 및 이미지]
                - 사용자가 향수를 사용할 수 있는 구체적인 상황, 라이프스타일, 감각적 연상 이미지와 감정적 연결을 한 문장으로 통합하여 설명합니다.
                - 예:
                    - "이 향수는 맑은 아침, 창문을 열었을 때 들어오는 깨끗한 공기처럼 상쾌한 느낌을 주며, 자신감과 차분함을 동시에 느낄 수 있어 업무 환경에서 집중력을 높이고 동료들에게 긍정적인 인상을 남길 수 있습니다."
                    - "햇살이 비치는 도심 속 카페의 모던한 인테리어를 떠올리게 하며, 설레는 기분과 부드러운 여유를 선사해 주말 데이트나 브런치에 잘 어울립니다."
                    
            - [향수의 사회적 이미지]
                - 3개의 향수들 공통점을 사용자가 어떻게 표현할 수 있는지, 사회적 이미지와의 연결성을 설명합니다.
                - 예:
                    - "이 향수들은 세련되고 신뢰감 있는 이미지를 전달하며, 직장이나 격식 있는 자리에서 돋보이는 존재감을 만듭니다."
                    - "자연스럽고 우아한 매력을 강조하여 사교적인 모임에서도 호감을 얻을 수 있습니다."
        """

        return template.format(user_input=user_input, perfumes_text=perfumes_text)

    def get_response(self, user_input: str, perfumes_text: str) -> str:
        """
        gpt-4를 사용하여 응답을 생성합니다.
        """
        try:
            prompt = self.create_prompt(user_input, perfumes_text)  # create_prompt 메서드를 호출
            return self.text_llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"gpt-4 호출 중 오류 발생: {str(e)}")
            return "죄송합니다. 요청 처리 중 문제가 발생했습니다."

    def recommend(self, user_input: str, perfumes_text: str) -> str:
        """
        사용자 입력을 기반으로 향수를 추천합니다.
        - user_input: 텍스트 입력
        - perfumes_text: 향수 데이터
        """
        if not user_input:
            raise ValueError("사용자 입력을 제공해야 합니다.")

        # gpt-4를 사용해 최종 추천 생성 (perfumes_text 전달)
        return self.get_response(user_input, perfumes_text)  # perfumes_text를 추가로 전달
