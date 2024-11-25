import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Optional
import openai

class GPTClient:
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # 환경 변수에서 API 키 가져오기
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        # 텍스트 처리용 ChatOpenAI 인스턴스 초기화 (gpt-4o-mini)
        self.text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

        # 이미지 처리용 OpenAI GPT-4 Vision API (gpt-4-vision)
        openai.api_key = api_key

    def process_image(self, image_data: bytes) -> str:
        """
        gpt-4-vision을 사용하여 이미지에서 텍스트를 추출하거나 분석합니다.
        """
        try:
            # 이미지 파일을 업로드하여 분석을 요청
            response = openai.Image.create(
                prompt="이미지 설명을 요청합니다.",
                n=1,
                images=[image_data]  # 이미지 데이터를 바이트 형태로 전달
            )

            # 결과 반환 (이미지에서 추출한 텍스트)
            return response['data'][0]['text']
        except Exception as e:
            print(f"gpt-4-vision 이미지 처리 중 오류 발생: {str(e)}")
            return ""  # 이미지 처리 결과가 없으면 빈 문자열 반환

    def create_prompt(self, user_input: str, perfumes_text: str) -> str:
        """
        gpt-4o-mini에 전달할 프롬프트를 생성합니다.
        """
        template = """
        당신은 향수 전문가입니다. 
        다음 조건에 맞는 응답을 답변하도록 하세요.
        추론되는 시간은 10초 이내로 하세요.
        이름에 오 드 뚜왈렛, 오 드 퍼퓸은 부향률이라는 뜻입니다. 이름에 표시하지 않고 부향률에 표시해주세요.
        조건: {user_input}

        향수 목록:
        {perfumes_text}

        **응답 예시**
        [추천계열] 추론된 계열이 무슨 계열과 매칭이 됐는지 어떤 것인지 명시, 부가 설명은 하지 않음
        [추천느낌] 추천된 향수들이 어떤 느낌을 주는지 설명해주세요.
        [추천향수] 조건과 가장 잘 맞는 3개의 향수를 추천해주세요. 
        이름: , 부향률 (예시) 오 드 퍼퓸, 퍼퓸, 오 드 뚜왈렛: , 브랜드: , 설명: , 추천이유:
        """
        return PromptTemplate(
            input_variables=["user_input", "perfumes_text"],
            template=template
        ).format(user_input=user_input, perfumes_text=perfumes_text)

    def get_response(self, user_input: str, perfumes_text: str) -> str:
        """
        gpt-4o-mini를 사용하여 응답을 생성합니다.
        """
        try:
            prompt = self.create_prompt(user_input, perfumes_text)
            return self.text_llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"gpt-4o-mini 호출 중 오류 발생: {str(e)}")
            return "죄송합니다. 요청 처리 중 문제가 발생했습니다."

    def recommend(self, user_input: str, image_data: bytes, perfumes_text: str) -> str:
        """
        사용자 입력과 이미지를 기반으로 향수를 추천합니다.
        - user_input: 텍스트 입력
        - image_data: 이미지 데이터
        - perfumes_text: 향수 데이터
        """
        # 이미지 데이터를 처리하여 설명 추출
        image_explanation = self.process_image(image_data) if image_data else ""

        # 사용자 입력과 이미지 설명을 결합
        combined_input = f"{user_input}\n이미지 설명: {image_explanation}" if image_explanation else user_input

        # user_input이 없고 image_explanation도 없으면 오류
        if not combined_input:
            raise ValueError("사용자 입력 또는 이미지 설명 중 하나가 필요합니다.")

        # gpt-4o-mini를 사용해 최종 추천 생성 (perfumes_text 전달)
        return self.get_response(combined_input, perfumes_text)
