from dotenv import load_dotenv
import logging, os
from langchain_openai import ChatOpenAI

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
load_dotenv()

class GPTClient:
    def __init__(self):  # prompt_loader 파라미터 제거
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_HOST")

        if not api_key:
            raise ValueError("🚨 OPENAI_API_KEY가 설정되지 않았습니다!")

        self.text_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base=api_base
        )

    async def generate_response(self, prompt: str) -> str:
        """GPT API를 호출하여 응답을 생성합니다."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.text_llm.ainvoke(prompt)  # ainvoke 사용
            return response.content
        except Exception as e:
            logger.error(f"GPT 응답 생성 실패: {e}")
            raise