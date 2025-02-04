from dotenv import load_dotenv
import logging, os
from services.prompt_loader import PromptLoader
from langchain_openai import ChatOpenAI

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
load_dotenv()

class GPTClient:
    def __init__(self, prompt_loader: PromptLoader):
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_HOST")  # ✅ 기본값 설정

        if not api_key:
            raise ValueError("🚨 OPENAI_API_KEY가 설정되지 않았습니다!")

        self.prompt_loader = prompt_loader

        # ✅ `openai_api_base` 추가하여 API 서버 주소 명확히 설정
        self.text_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base=api_base  # ✅ API 주소 설정
        )

    def generate_response(self, prompt: str) -> str:
        try:
            logger.info(f"🔹 Generating response for prompt: {prompt}...")

            response = self.text_llm.invoke(prompt).content.strip()

            logger.info(f"✅ Generated response: {response}...")
            return response
        except Exception as e:
            logger.error(f"🚨 GPT 응답 생성 오류: {e}")
            raise RuntimeError("🚨 GPT 응답 생성 오류")
