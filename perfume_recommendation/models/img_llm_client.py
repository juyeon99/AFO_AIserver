from dotenv import load_dotenv
import logging, os
from services.prompt_loader import PromptLoader
from services.db_service import DBService
from langchain_openai import ChatOpenAI

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
load_dotenv()


class GPTClient:
    def __init__(self, template_path: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.db_service = DBService()
        self.prompt_loader = PromptLoader(template_path)

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        # GPT-4o-mini 초기화
        self.text_llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.9, openai_api_key=self.api_key
        )

    def generate_response(self, prompt: str) -> str:
        """
        GPT 모델을 호출하여 프롬프트 기반으로 응답을 생성합니다.
        """
        try:
            response = self.text_llm.invoke(prompt).content.strip()
            return response
        except Exception as e:
            logger.error(f"GPT 응답 생성 오류: {e}")
            raise RuntimeError("GPT 응답 생성 오류")