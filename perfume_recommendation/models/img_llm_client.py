import os
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_openai import ChatOpenAI
from PIL import Image
from io import BytesIO

load_dotenv()

class GPTClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        try:
            self.text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=self.api_key)
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        except Exception as e:
            raise RuntimeError(f"모델 초기화 실패: {e}")

    def process_image(self, image_data: bytes) -> str:
        """
        이미지 데이터를 처리하여 설명을 생성하고 필요시 요약합니다.
        """
        try:
        # 이미지를 로드하고 정보 출력
            image = Image.open(BytesIO(image_data))
            print(f"이미지 포맷: {image.format}")
            print(f"이미지 크기: {image.size}")
            print(f"이미지 모드: {image.mode}")

        # 이미지 처리
            inputs = self.processor(image, return_tensors="pt")

        # 토큰 제한을 설정하고 설명 생성
            max_new_tokens = 50
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # 생성된 토큰 수 계산
            generated_tokens = outputs[0].size(-1)  # 생성된 텐서의 마지막 차원이 토큰 수
            print(f"생성된 토큰 수: {generated_tokens} / 최대 허용 토큰: {max_new_tokens}")

        # 설명 디코딩
            description = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

        # 설명이 길면 요약
            if len(description.split()) > max_new_tokens:
                description = self.summarize_description(description)

            return description
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            return "이미지에서 설명을 생성할 수 없습니다."


    def summarize_description(self, description: str) -> str:
        """
        설명이 너무 길 경우 요약합니다.
        """
        prompt = f"다음 설명을 간략하게 요약해 주세요: {description}"
        try:
            response = self.text_llm.invoke(prompt).content.strip()
            return response or "설명을 요약할 수 없습니다."
        except Exception as e:
            print(f"GPT-4 호출 중 오류 발생: {str(e)}")
            return "설명을 요약할 수 없습니다."

    def generate_image_feeling(self, image_description: str) -> str:
        """
        GPT-4를 사용하여 이미지 느낌을 생성합니다.
        """
        if not image_description:
            return "이미지 설명이 제공되지 않았습니다."

        prompt = f"이 이미지는 '{image_description}'을(를) 보여줍니다. 이 이미지가 주는 느낌을 간략히 설명해주세요."
        try:
            response = self.text_llm.invoke(prompt).content.strip()
            return response or "이미지 느낌을 생성할 수 없습니다."
        except Exception as e:
            print(f"GPT-4 호출 중 오류 발생: {str(e)}")
            return "이미지 느낌을 분석할 수 없습니다."

    def create_prompt(self, user_input: str, perfumes_text: str) -> str:
        """
        GPT-4 호출에 사용할 프롬프트를 생성합니다.
        """
        # 필요 없는 설명이나 반복 제거
        template = """
        당신은 향수 전문가입니다.
        다음 조건에 맞는 향수를 추천해주세요.

        [조건]
        {user_input}

        [향수 목록]
        {perfumes_text}

        [응답 형식]
        1. [추천 계열]: 추천 향수 계열 명시.
        2. [추천 느낌]: 향수 느낌 설명.
        3. [추천 향수]: 적합한 향수 3개 제안.
        """
        return template.format(user_input=user_input, perfumes_text=perfumes_text)

    def get_response(self, combined_input: str, perfumes_text: str) -> str:
        """
        GPT-4를 사용하여 향수 추천을 생성합니다.
        """
        prompt = self.create_prompt(combined_input, perfumes_text)
        try:
            response = self.text_llm.invoke(prompt).content.strip()
            return response or "추천 결과를 생성할 수 없습니다."
        except Exception as e:
            print(f"GPT-4 호출 중 오류 발생: {str(e)}")
            return "죄송합니다. 추천을 생성할 수 없습니다."
