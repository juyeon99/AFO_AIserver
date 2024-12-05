import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
load_dotenv()

class ImageGenerationService:
    def __init__(self):
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.image_folder = os.getenv("IMAGE_FOLDER", "generated_images")  # ./ 제거
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000")
        os.makedirs(self.image_folder, exist_ok=True)

        if not self.stability_api_key:
            raise ValueError("STABILITY_API_KEY 환경 변수가 설정되지 않았습니다.")

    def generate_image(self, imageGeneratePrompt: str) -> dict:
        try:
            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "image/*"
            }
            files = {
                "imageGeneratePrompt": (None, imageGeneratePrompt),
                "output_format": (None, "jpeg"),
            }

            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers=headers,
                files=files
            )

            if response.status_code == 200:
                output_filename = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"
                
                # 파일 저장 경로
                output_path = Path(self.image_folder) / output_filename
                
                # 이미지 저장
                with open(output_path, "wb") as file:
                    file.write(response.content)

                # URL 경로 생성 - /static/ 다음에 바로 파일명이 오도록 수정
                relative_url = f"/static/{output_filename}"
                
                logger.info(f"이미지가 성공적으로 생성되었습니다: {output_path}")
                return {
                    "output_path": relative_url,
                    "absolute_path": str(output_path.resolve())
                }
            else:
                try:
                    error_details = response.json()
                except ValueError:
                    error_details = response.text
                logger.error(f"이미지 생성 실패: {error_details}")
                raise ValueError(f"이미지 생성 오류: {error_details}")

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}")
            raise ValueError(f"이미지 생성 중 오류 발생: {str(e)}")