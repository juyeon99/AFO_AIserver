import logging
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageProcessingService:
    def __init__(self):
        # BLIP 모델 초기화
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            logger.info("BLIP 모델이 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"BLIP 모델 초기화 실패: {str(e)}")
            raise RuntimeError("BLIP 모델 초기화 중 문제가 발생했습니다.")

    def download_image_from_url(self, image_url: str) -> bytes:
        """
        이미지 URL에서 이미지를 다운로드하여 바이트 데이터를 반환합니다.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": image_url,
            }
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            return response.content
        except requests.exceptions.RequestException as e:
            raise ValueError(f"이미지 다운로드 실패: {e}")

    def process_image(self, image_data: bytes) -> dict:
        """
        이미지 데이터를 받아 설명(description)과 느낌(feeling)을 생성합니다.
        """
        try:
            # 이미지 로드
            image = Image.open(BytesIO(image_data))

            # 이미지 설명 생성
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            description = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            # 느낌 생성 (간단히 설명에서 추론)
            feeling = f"이 이미지는 '{description}'을(를) 보여주며, 평화롭고 감각적인 분위기를 전달합니다."

            return {"description": description, "feeling": feeling}
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
            raise ValueError("이미지 처리 중 오류가 발생했습니다.")
