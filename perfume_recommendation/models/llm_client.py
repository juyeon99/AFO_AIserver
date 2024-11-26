import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
import logging

logger = logging.getLogger(__name__)

class GPTClient:
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # 환경 변수에서 API 키 가져오기
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 안정성 API 키도 확인 (필요한 경우)
        if not self.stability_api_key:
            raise ValueError("STABILITY_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 텍스트 처리용 ChatOpenAI 인스턴스 초기화
        self.text_llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=self.api_key)

        # 프롬프트 파일 경로 지정
        self.prompt_template_path = "prompt_template.txt"
        
        # 이미지 저장 폴더 경로
        self.image_folder = "image"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def generate_image(self, prompt: str, output_filename: str) -> None:
        """이미지 생성 API 호출하여 이미지를 생성합니다 (Stability.ai)."""
        try:
            # 요청 본문에 필요한 데이터를 json 형식으로 보내기
            data = {
                'prompt': prompt,
                'output_format': 'jpeg',  # 지원하는 형식에 맞춰 수정
            }

            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "application/json"
            }

            # API 요청 (파일 업로드 또는 json 사용 여부 확인)
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers=headers,
                json=data  # JSON 형식으로 전달
            )

            if response.status_code == 200:
                output_path = os.path.join(self.image_folder, output_filename)
                with open(output_path, 'wb') as file:
                    file.write(response.content)
                print(f"이미지가 성공적으로 생성되었습니다: {output_path}")
            else:
                logger.error(f"이미지 생성 오류: {response.json()}")
                raise Exception(f"이미지 생성 오류: {response.json()}")
        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}")
            print(f"이미지 생성 중 오류 발생: {str(e)}")

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
        """
        gpt-4를 사용하여 응답을 생성합니다.
        """
        try:
            prompt = self.create_prompt(user_input, perfumes_text)  # create_prompt 메서드를 호출
            return self.text_llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.error(f"gpt-4 호출 중 오류 발생: {str(e)}")
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
