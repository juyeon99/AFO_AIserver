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
        self.text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=self.api_key)

    def process_user_input(self, user_input: str) -> dict:
        """
        사용자의 입력에 따라 대화 또는 향수 추천을 결정하고 응답을 생성합니다.
        """
        try:
            # 대화와 추천 구분
            if "안녕" in user_input or "대화" in user_input:
                prompt = self.prompt_loader.get_prompt("chat")["example_prompt"]
                final_prompt = f"{prompt}\n{user_input}"
                response = self.text_llm.invoke(final_prompt).content.strip()
                return {"mode": "chat", "response": response}
            
            elif "향수" in user_input or any(keyword in user_input for keyword in ["머스크", "시트러스", "우디"]):
                perfumes = self.db_service.fetch_perfume_data()
                if not perfumes:
                    raise ValueError("향수 데이터를 가져올 수 없습니다.")
                
                # 추천 프롬프트 생성
                template = self.prompt_loader.get_prompt("recommendation")
                perfumes_text = "\n".join(
                    [f"{perfume['name']}: {perfume['description']}" for perfume in perfumes]
                )
                final_prompt = template["example_prompt"].format(user_input=user_input, perfumes_text=perfumes_text)
                recommendation = self.text_llm.invoke(final_prompt).content.strip()

                # 공통 느낌 생성
                common_feeling_prompt = template["common_feeling_prompt"].format(recommendation=recommendation)
                common_feeling = self.text_llm.invoke(common_feeling_prompt).content.strip()

                return {
                    "mode": "recommendation",
                    "recommended_perfumes": recommendation,
                    "common_feeling": common_feeling
                }
            else:
                raise ValueError("적절한 입력을 제공해주세요.")
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return {"error": str(e)}
        
    def get_response(self, prompt: str, context: str = "") -> str:
        """
        GPT 모델을 호출하여 응답을 생성합니다.
        """
        # GPT 모델과의 통신 로직 구현
        # 아래는 예시
        try:
            # 예시: GPT API 호출
            response = f"Generated response for prompt: {prompt} with context: {context}"
            return response
        except Exception as e:
            raise RuntimeError(f"GPT 호출 실패: {str(e)}")

