import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from io import BytesIO

# OpenMP 충돌 방지 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

class ImageProcessingService:
    def __init__(self):
        """Florence-2 모델 및 프로세서를 초기화"""
        try:
            print("🔹 Florence-2 모델 로드 중...")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # 모델 및 프로세서 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True
            )

            print("✅ Florence-2 모델 로드 완료!")
        
        except Exception as e:
            print(f"🚨 모델 초기화 중 오류 발생: {e}")
            raise RuntimeError("🚨 모델을 불러오는 중 오류 발생!")

    def process_image(self, image_data: bytes) -> dict:
        """이미지에서 텍스트 설명을 생성"""
        try:
            print("🔹 이미지 처리 중...")
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = image.resize((512, 512))

            # 프롬프트 설정
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # 장치 및 데이터 타입 변환
            inputs["input_ids"] = inputs["input_ids"].to(self.device, dtype=torch.long)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device, dtype=torch.float16)

            # 모델 예측
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=5,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )

            # 텍스트 디코딩
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("✅ 생성된 설명:", description)

            # 간단한 감정 분석 추가 (예시)
            if "happy" in description.lower() or "cheerful" in description.lower():
                feeling = "Positive"
            elif "dark" in description.lower() or "sad" in description.lower():
                feeling = "Negative"
            else:
                feeling = "Neutral"

            return {"description": description, "feeling": feeling}

        except Exception as e:
            print(f"🚨 이미지 처리 중 오류 발생: {e}")
            return {"error": f"🚨 이미지 처리 실패: {str(e)}"}
