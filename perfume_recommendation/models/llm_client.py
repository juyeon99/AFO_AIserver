import os
import boto3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
import logging
from botocore.exceptions import ClientError
import io
import base64  # base64 모듈 추가

logger = logging.getLogger(__name__)

class GPTClient:
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # 환경 변수에서 API 키와 AWS 자격 증명 가져오기
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("BUCKET_NAME")
        self.region = os.getenv("AWS_REGION")

        # prompt 템플릿 경로 설정
        self.prompt_template_path = "prompt_template.txt"  # 또는 전체 경로를 지정

        # 필수 자격 증명 검증
        if not all([self.api_key, self.stability_api_key, 
                    self.aws_access_key, self.aws_secret_key, self.bucket_name, self.region]):
            raise ValueError("필수 환경 변수가 설정되지 않았습니다.")
        
        # S3 클라이언트 초기화
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )
        

        # ChatOpenAI 인스턴스 초기화
        self.text_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.7, 
            openai_api_key=self.api_key
        )
        
    def translate_to_english(self, korean_text: str) -> str:
        """한국어 텍스트를 영어로 번역"""
        try:
            prompt = f"Translate the following Korean text to English, focusing on fragrance description: {korean_text}"
            translation = self.text_llm.invoke(prompt).content.strip()
            return translation
        except Exception as e:
            logger.error(f"번역 중 오류 발생: {str(e)}")
            return "elegant and sophisticated atmosphere"

    def generate_and_upload_image(self, prompt: str, s3_key: str) -> str:
        try:
            data = {
                "text_prompts": [{"text": prompt}],
                "steps": 30,
                "width": 1024,
                "height": 1024,
                "samples": 1,
                "cfg_scale": 7
            }

            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            response = requests.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                raise Exception(f"이미지 생성 실패: {response.text}")

            # 응답에서 이미지 데이터 추출
            image_data = response.json()['artifacts'][0]['base64']
            image_bytes = io.BytesIO(base64.b64decode(image_data))  # 수정된 부분
            
            # S3에 이미지 업로드
            self.s3_client.upload_fileobj(
                image_bytes,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png' }
            )

            # S3 URL 생성
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            return s3_url

        except Exception as e:
            logger.error(f"이미지 생성 및 업로드 중 오류 발생: {str(e)}")
            raise

    def load_prompt_template(self) -> str:
        """프롬프트 템플릿을 외부 파일에서 로드합니다."""
        try:
            with open(self.prompt_template_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"프롬프트 템플릿 로드 오류: {str(e)}")
            print(f"프롬프트 템플릿 로드 오류: {str(e)}")
            return ""

    def create_prompt(self, user_input: str, perfumes_text: str) -> str:
        """
        gpt-4에 전달할 프롬프트를 생성합니다.
        """
        template = self.load_prompt_template()
        if not template:
            raise ValueError("프롬프트 템플릿을 로드할 수 없습니다.")
        
        return template.format(user_input=user_input, perfumes_text=perfumes_text)

    def get_response(self, user_input: str, perfumes_text: str) -> str:
            """GPT-4를 사용하여 응답을 생성합니다."""
            try:
                prompt = self.create_prompt(user_input, perfumes_text)
                return self.text_llm.invoke(prompt).content.strip()
            except Exception as e:
                logger.error(f"GPT-4 호출 중 오류 발생: {str(e)}")
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
    
    
