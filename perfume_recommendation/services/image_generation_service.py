import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
load_dotenv()

class ImageGenerationService:
    def __init__(self):
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.image_folder = os.getenv("IMAGE_FOLDER", "./generated_images")
        os.makedirs(self.image_folder, exist_ok=True)

        if not self.stability_api_key:
            raise ValueError("STABILITY_API_KEY 환경 변수가 설정되지 않았습니다.")

    def generate_image(self, prompt: str) -> str:
        """
        텍스트 프롬프트를 기반으로 이미지를 생성하고 저장합니다.
        """
        try:
            # Stable Diffusion API 요청
            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "image/*"  # 이미지를 직접 반환받기 위해 설정
            }
            # Multipart/form-data 요청 데이터 구성
            files = {
                "prompt": (None, prompt),  # 프롬프트 텍스트
                "output_format": (None, "jpeg"),  # 출력 형식
            }

            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers=headers,
                files=files  # multipart/form-data 형식으로 데이터 전송
            )

            if response.status_code == 200:
                # 이미지 파일 저장
                output_filename = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"
                output_path = os.path.join(self.image_folder, output_filename)
                with open(output_path, "wb") as file:
                    file.write(response.content)

                logger.info(f"이미지가 성공적으로 생성되었습니다: {output_path}")
                return output_path
            else:
                # 실패한 경우, 오류 세부 정보 출력
                try:
                    error_details = response.json()
                except ValueError:
                    error_details = response.text  # JSON이 아닐 경우 텍스트 처리
                logger.error(f"이미지 생성 실패: {error_details}")
                raise ValueError(f"이미지 생성 오류: {error_details}")

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}")
            raise ValueError(f"이미지 생성 중 오류 발생: {str(e)}")
