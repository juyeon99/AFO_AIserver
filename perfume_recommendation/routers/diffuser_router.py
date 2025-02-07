import os
from fastapi import APIRouter, Depends, HTTPException
from services.diffuser_service import DiffuserRecommendationService
from services.db_service import DBService
from models.client import GPTClient
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# from dotenv import find_dotenv, load_dotenv
# import os

# # .env 파일 위치 확인
# env_path = find_dotenv()
# print(f"Found .env at: {env_path}")

# # .env 파일 내용 확인
# with open(env_path, 'r', encoding='utf-8') as f:
#     print("=== .env 파일 내용 ===")
#     print(f.read())

# # 기존 환경 변수 값들 확인
# print("\n=== 현재 환경 변수 값 ===")
# for key, value in os.environ.items():
#     if 'DB_' in key:
#         print(f"{key}: {value}")

# # 환경 변수 강제 재설정
# load_dotenv(override=True)

def get_diffuser_service() -> DiffuserRecommendationService:
    try:
        
        
        db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT")),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
        
        # 설정값 로깅
        logger.debug(f"DB HOST: {db_config['host']}")
        logger.debug(f"DB PORT: {db_config['port']}")
        logger.debug(f"DB USER: {db_config['user']}")
        logger.debug(f"DB NAME: {db_config['database']}")
        
        if not all(db_config.values()):
            raise RuntimeError("데이터베이스 설정이 불완전합니다.")

        # GPTClient 생성 시 파라미터 없이 초기화
        gpt_client = GPTClient()  # api_key 파라미터 제거
        db_service = DBService(db_config=db_config)

        return DiffuserRecommendationService(gpt_client=gpt_client, db_service=db_service)
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise

@router.post("/recommend")
async def recommend_diffusers(
    category: str,
    diffuser_service: DiffuserRecommendationService = Depends(get_diffuser_service)
) -> dict:
    """디퓨저 추천 엔드포인트"""
    try:
        result = await diffuser_service.recommend_diffusers(category)
        return result
    except Exception as e:
        logger.error(f"추천 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))