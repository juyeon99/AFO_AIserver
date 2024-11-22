from typing import BinaryIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageService:
    def __init__(self):
        # BLIP 모델 초기화
        self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def process_image(self, file: BinaryIO) -> str:
        """이미지에서 텍스트를 추출합니다."""
        try:
            image = Image.open(file).convert("RGB")
            inputs = self.image_processor(image, return_tensors="pt")
            caption = self.image_model.generate(**inputs)
            return self.image_processor.decode(caption[0], skip_special_tokens=True)
        except Exception as e:
            raise ValueError(f"이미지 처리 중 오류 발생: {str(e)}")
