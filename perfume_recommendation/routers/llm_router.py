import os
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from services.llm_service import LLMService
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from models.img_llm_client import GPTClient
import logging

logger = logging.getLogger(__name__)

# FastAPI 인스턴스 생성
app = FastAPI()

# Router 생성
router = APIRouter()

# 의존성 주입 함수
def get_llm_service() -> LLMService:
    try:
        # 경로 설정
        base_path = os.path.abspath(os.path.dirname(__file__))
        template_path = os.path.join(base_path,"..","models","chat_prompt_template.json")

        # 파일 확인
        if not os.path.exists(template_path):
            raise RuntimeError(f"템플릿 파일을 찾을 수 없습니다: {template_path}")

        # 환경 변수 로드
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT")),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
        if not all(db_config.values()):
            raise RuntimeError("데이터베이스 설정이 불완전합니다. 환경 변수를 확인하세요.")

        # 서비스 초기화
        db_service = DBService(db_config=db_config)
        prompt_loader = PromptLoader(template_path)
        gpt_client = GPTClient(prompt_loader=prompt_loader)

        return LLMService(gpt_client=gpt_client, db_service=db_service, prompt_loader=prompt_loader)
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise

# 라우터에 엔드포인트 추가
@router.post("/process-input")
async def process_input(input_data: dict, llm_service: LLMService = Depends(get_llm_service)):
    """
    사용자 입력 처리 및 대화/추천 결과 반환
    """
    try:
        user_input = input_data["user_input"]
        mode, response = llm_service.process_input(user_input)

        logger.info(f"사용자 입력 처리: mode={mode}, input={user_input}")

        if mode == "chat":
            return {"mode": "chat", "content": response}

        elif mode == "recommendation":
            recommendations = response.get("recommendations", [])
            content = response.get("content", "공통 감정 생성 실패")
            line_id = response.get("line_id", "line_id 생성 실패")

            return {
                "mode": "recommendation", 
                "recommendations": recommendations,
                "content": content,
                "lineId": line_id,
            }

        else:
            raise HTTPException(status_code=400, detail="알 수 없는 모드")

    except HTTPException as e:
        logger.error(f"HTTP 예외 발생: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
