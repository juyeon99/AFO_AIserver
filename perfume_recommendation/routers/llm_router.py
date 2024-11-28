from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.llm_service import LLMService
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from models.img_llm_client import GPTClient
import os

# 라우터 인스턴스 생성
router = APIRouter()

# 필요한 의존성 초기화
template_path = os.path.join(os.path.dirname(__file__), "..", "models", "prompt_template.json")
prompt_loader = PromptLoader(template_path)
gpt_client = GPTClient(template_path=template_path)
db_service = DBService()

# LLMService 인스턴스 생성
llm_service = LLMService(gpt_client=gpt_client, db_service=db_service, prompt_loader=prompt_loader)

# Pydantic 모델 정의
class UserInput(BaseModel):
    user_input: str


# 엔드포인트 정의
@router.post("/process-input")
async def process_user_input(input_data: UserInput):
    """
    사용자 입력을 처리하여 대화 또는 향수 추천을 반환합니다.
    """
    try:
        user_input = input_data.user_input
        mode, data = llm_service.process_input(user_input)

        if mode == "chat":
            response = llm_service.generate_chat_response(user_input)
            return {"mode": "chat", "response": response}

        elif mode == "recommendation":
            perfumes = data
            response = llm_service.generate_recommendation_response(user_input, perfumes)
            return {
                "mode": "recommendation",
                "recommended_perfumes": response["recommendation"],
                "common_feeling": response["common_feeling"],
            }

        else:
            raise HTTPException(status_code=400, detail="Unknown mode")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Input Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
