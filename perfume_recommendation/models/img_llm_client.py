from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from PIL import Image
from io import BytesIO
import logging , requests , os 
from transformers import BlipProcessor, BlipForConditionalGeneration

# 로거 설정
logger = logging.getLogger(__name__)

load_dotenv()

class GPTClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.stability_api_key = os.getenv("STABILITY_API_KEY")  
        self.image_folder = os.getenv("IMAGE_FOLDER", "./perfume_recommendation/images") 
        os.makedirs(self.image_folder, exist_ok=True)
        self.prompt_template_path = "./models/prompt_template.txt"

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        try:
            self.text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=self.api_key)
            # BLIP 모델 초기화
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        except Exception as e:
            raise RuntimeError(f"모델 초기화 실패: {e}")
    
    def generate_image(self, prompt: str, output_filename: str) -> None:
        """
        Stability.ai API를 사용하여 이미지를 생성하고 지정된 폴더에 저장합니다.
        프롬프트와 사용자 입력을 결합하여 이미지를 생성합니다.
        """
        try:
            # Stability.ai API 요청 데이터 (multipart/form-data)
            data = {
                "prompt": prompt,
                "output_format": "jpeg",  # 출력 형식
            }

            files = {
                "file": (output_filename, open(output_filename, 'rb'), "image/jpeg")
            }

            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "application/json"
            }

            # Stability.ai API 요청
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers=headers,
                files=files,  # files는 multipart/form-data 형식으로 전송
                data=data  # JSON 데이터를 'data'로 전달
            )

            if response.status_code == 200:
                # 이미지 URL을 추출하고 다운로드
                result = response.json()
                image_url = result.get("image_url")
                if not image_url:
                    raise Exception("이미지 URL이 응답에 포함되지 않았습니다.")
                
                image_response = requests.get(image_url, timeout=10)
                image_response.raise_for_status()

                # 이미지 저장
                output_path = os.path.join(self.image_folder, output_filename)
                with open(output_path, "wb") as file:
                    file.write(image_response.content)

                print(f"이미지가 성공적으로 생성되었습니다: {output_path}")
            else:
                logger.error(f"이미지 생성 오류: {response.json()}")
                raise Exception(f"이미지 생성 오류: {response.json()}")

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}")
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            
    def process_image(self, image_data: bytes) -> str:
        """
        이미지를 처리하여 설명을 생성합니다.
        """
        try:
            # 이미지 로드
            image = Image.open(BytesIO(image_data))

            # BLIP 모델을 사용해 이미지 설명 생성
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            description = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            return description
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
            return "이미지 설명을 생성할 수 없습니다."

    def summarize_image_url(self, image_url: str) -> str:
        """
        이미지 URL을 기반으로 설명을 요약합니다.
        """
        if not image_url:
            return "이미지 URL이 제공되지 않았습니다."

        prompt = f"""
        주어진 이미지 URL은 '{image_url}'입니다.
        이 URL이 표현하는 이미지의 내용을 간단히 추측하고 요약해주세요.
        URL 자체에 포함된 내용이나 연관된 이미지의 특징을 기반으로 작성하세요.
        """
        try:
            response = self.text_llm.invoke(prompt).content.strip()
            return response or "이미지 URL에서 설명을 생성할 수 없습니다."
        except Exception as e:
            print(f"GPT-4 호출 중 오류 발생: {str(e)}")
            return "이미지 URL 설명을 생성할 수 없습니다."

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
