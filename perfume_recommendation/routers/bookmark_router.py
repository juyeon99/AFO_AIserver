from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from services.db_service import get_db
from services.bookmark_service import PerfumeRecommender
from services.mongo_service import MongoService

router = APIRouter()

# MongoService 인스턴스를 제공하는 의존성 함수
async def get_mongo_service():
    mongo_service = MongoService()
    try:
        yield mongo_service
    finally:
        del mongo_service  # 연결 종료 보장

@router.get("/recommendations/{member_id}")
async def get_recommendations(
    member_id: int,
    limit: int = 5,
    db: Session = Depends(get_db),
    mongo_service: MongoService = Depends(get_mongo_service)
):
    try:
        recommender = PerfumeRecommender(mongo_service)
        recommendations = recommender.get_recommendations(member_id, db, top_n=limit)
        return recommendations
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )