from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class ImageProcessingService:
    def __init__(self):
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            self.chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 이미지 설명을 감성적이고 어떤 느낌이 드는지 바꿔주는 전문가입니다."),
                ("user", "다음 이미지 설명을 더 감성적이고 어떤 느낌이 들고 어떤 향이 어울릴지 추천받고 향수를 찾아다라고를 한줄로 바꿔주세요: '{description}'")
            ])
        except Exception as e:
            raise RuntimeError(f"모델 초기화 실패: {e}")

    def get_emotional_caption(self, description: str) -> str:
        try:
            chain = self.prompt | self.chat
            result = chain.invoke({"description": description})
            return result.content
        except Exception as e:
            print(f"GPT 처리 중 오류 발생: {e}")
            return description

    def process_image(self, image_data: bytes) -> dict:
        try:
            image = Image.open(BytesIO(image_data))

            # max_new_tokens 파라미터 추가
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # 생성할 최대 토큰 수 지정
                min_new_tokens=20,  # 최소 토큰 수도 지정하여 더 자세한 설명 유도
                do_sample=True,     # 다양한 설명 생성 가능하도록 설정
                temperature=0.5,    # 창의성 조절 (0.0~1.0)
                num_beams=5,        # 빔 서치를 통한 더 나은 설명 생성
                no_repeat_ngram_size=2  # 반복 구문 방지
            )
            description = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            feeling = self.get_emotional_caption(description)

            return {
                "description": description,
                "feeling": feeling
            }
        except Exception as e:
            raise ValueError(f"이미지 처리 중 오류 발생: {e}")