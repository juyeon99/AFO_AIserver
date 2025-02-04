import os
import torch
import traceback
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
import openai  
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from io import BytesIO

# OpenMP 충돌 방지 및 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ImageProcessingService:
    def __init__(self):
        try:
            print("🔹 Florence 모델 로드 중...")
            # Florence 모델과 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large", trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large", trust_remote_code=True, torch_dtype=torch.float16
            )
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            if self.device == "cpu":
                print("🚨 CUDA 사용 불가능! CPU로 실행합니다.")
            else:
                print("✅ CUDA 사용 가능, GPU로 모델 실행!")

            self.model.to(self.device)
            print("✅ Florence 모델 로드 완료!")

            # OpenAI GPT 모델 초기화
            print("🔹 OpenAI GPT 모델 로드 중...")
            openai_api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
            if not openai_api_key:
                raise ValueError("🚨 OpenAI API Key가 설정되지 않았습니다. 환경 변수를 확인하세요.")

            self.chat = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=openai_api_key)
            self.root_client = openai.OpenAI(api_key=openai_api_key)  

            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 이미지 설명을 감성적이고 어떤 느낌이 드는지 바꿔주는 전문가입니다."),
                ("user", "다음 이미지 설명을 더 감성적이고 어떤 느낌이 들고 어떤 향이 어울릴지 추천받고 향수를 찾아달라고를 한 줄로 바꿔주세요: '{description}'")
            ])
            print("✅ OpenAI GPT 모델 로드 완료!")
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"🚨 모델 초기화 실패: {e}\n{error_trace}")
            raise RuntimeError(f"🚨 모델 초기화 실패: {e}")

    def get_emotional_caption(self, description: str) -> str:
        """ GPT를 이용해 감성적인 이미지 설명 생성 """
        try:
            chain = self.prompt | self.chat
            result = chain.invoke({"description": description})
            return result.content
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"🚨 GPT 처리 중 오류 발생: {e}\n{error_trace}")
            return description

    def process_image(self, image_data: bytes) -> dict:
        """ 이미지를 처리하여 캡션을 생성하고 감성적인 설명으로 변환 """
        try:
            print("🔹 이미지 처리 중...")
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = image.resize((512, 512))

            # Florence 모델을 사용하여 설명 생성
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # ✅ 모든 입력 데이터를 self.device로 이동 (오류 방지)
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16, device=self.device)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,  # ✅ 메모리 초과 방지를 위해 토큰 수 줄임
                num_beams=5,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )

            # 생성된 텍스트 디코딩
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("✅ Florence 모델 생성 결과:", generated_text)

            # GPT를 사용하여 감성적인 설명 생성
            description = generated_text
            feeling = self.get_emotional_caption(description)

            return {
                "description": description,
                "feeling": feeling
            }
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"🚨 이미지 처리 중 오류 발생: {e}\n{error_trace}")
            raise RuntimeError(f"🚨 이미지 처리 중 오류 발생: {e}")

    def draw_boxes(self, image, results):
        """ 이미지 위에 바운딩 박스를 그리고 라벨을 표시 """
        draw = ImageDraw.Draw(image)
        bboxes = results.get("<DENSE_REGION_CAPTION>", {}).get("bboxes", [])
        labels = results.get("<DENSE_REGION_CAPTION>", {}).get("labels", [])

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), label, fill="red")

        return image