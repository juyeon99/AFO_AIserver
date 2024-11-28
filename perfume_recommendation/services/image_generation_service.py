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
                "Accept": "application/json"
            }
            data = {
                "prompt": prompt,
                "output_format": "jpeg"
            }
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                image_url = result.get("image_url")
                if not image_url:
                    raise ValueError("API 응답에 이미지 URL이 포함되지 않았습니다.")

                # 이미지 다운로드 및 저장
                image_response = requests.get(image_url, timeout=10)
                image_response.raise_for_status()

                output_filename = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"
                output_path = os.path.join(self.image_folder, output_filename)
                with open(output_path, "wb") as file:
                    file.write(image_response.content)

                logger.info(f"이미지가 성공적으로 생성되었습니다: {output_path}")
                return output_path
            else:
                logger.error(f"이미지 생성 실패: {response.json()}")
                raise ValueError(f"이미지 생성 오류: {response.json()}")
        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}")
            raise ValueError(f"이미지 생성 중 오류 발생: {str(e)}")
