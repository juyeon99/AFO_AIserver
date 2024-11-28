from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO

class ImageProcessingService:
    def __init__(self):
        # BLIP 모델과 프로세서를 초기화
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        except Exception as e:
            raise RuntimeError(f"BLIP 모델 초기화 실패: {e}")

    def process_image(self, image_data: bytes) -> dict:
        """
        업로드된 이미지 데이터를 처리하여 설명과 느낌을 반환합니다.
        """
        try:
            # 이미지 로드
            image = Image.open(BytesIO(image_data))

            # BLIP 모델을 사용하여 이미지 설명 생성
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            description = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            # 예제 느낌 생성 (BLIP 모델 기반 또는 단순 논리)
            feeling = f"이미지 해석 '{description}'"

            return {"description": description, "feeling": feeling}
        except Exception as e:
            raise ValueError(f"이미지 처리 중 오류 발생: {e}")
