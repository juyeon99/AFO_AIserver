from fastapi import APIRouter, HTTPException
from services.recommendation_service import RecommendationService

router = APIRouter()
service = RecommendationService()

@router.get("/list")
async def list_recommendations():
    try:
        return service.get_all_recommendations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 사용자 입력을 받아 추천할 향수를 반환하는 API
@router.post("/recommend")
async def recommend_perfume(description: str):
    recommender = RecommendationService()
    if not recommender.load_data():
        raise HTTPException(status_code=500, detail="향수 데이터를 로드하는데 실패했습니다.")
    
    try:
        recommendation = recommender.recommend_perfumes(description)
        return {"description": description, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 과정 중 오류가 발생했습니다: {str(e)}")

