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

logger = logging.getLogger(__name__)

class GPTClient:
    def __init__(self, prompt_loader: PromptLoader):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.prompt_loader = prompt_loader

        # GPT-4 모델 초기화
        self.text_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key
        )

    def generate_response(self, prompt: str) -> str:
        try:
            # 여기에서만 필요한 경우 import
            from services.db_service import DBService
            # 실제로 DBService를 사용할 때 가져옵니다.
            db_service = DBService({"some": "config"})
            
            logger.info(f"Generating response for prompt: {prompt}...")
            response = self.text_llm.invoke(prompt).content.strip()
            logger.info(f"Generated response: {response}...")
            return response
        except Exception as e:
            logger.error(f"GPT 응답 생성 오류: {e}")
            raise RuntimeError("GPT 응답 생성 오류")