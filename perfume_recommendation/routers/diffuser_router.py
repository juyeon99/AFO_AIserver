import os
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.diffuser_service import DiffuserRecommendationService
from services.db_service import DBService
from models.client import GPTClient
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# 요청 바디 모델 정의
class DiffuserRecommendRequest(BaseModel):
    user_input: str

def get_diffuser_service() -> DiffuserRecommendationService:
    try:
        db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT")),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
        
        logger.debug(f"DB HOST: {db_config['host']}")
        logger.debug(f"DB PORT: {db_config['port']}")
        logger.debug(f"DB USER: {db_config['user']}")
        logger.debug(f"DB NAME: {db_config['database']}")
        
        if not all(db_config.values()):
            raise RuntimeError("데이터베이스 설정이 불완전합니다.")

        gpt_client = GPTClient()
        db_service = DBService(db_config=db_config)

        return DiffuserRecommendationService(gpt_client=gpt_client, db_service=db_service)
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise

@router.post("/recommend")
async def recommend_diffusers(
    request: DiffuserRecommendRequest,
    diffuser_service: DiffuserRecommendationService = Depends(get_diffuser_service)
) -> dict:
    """디퓨저 추천 엔드포인트"""
    try:
        result = await diffuser_service.recommend_diffusers(request.user_input)
        return result
    except Exception as e:
        logger.error(f"추천 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))