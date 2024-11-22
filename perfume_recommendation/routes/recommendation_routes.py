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
    """
    사용자 입력(description)을 기반으로 향수를 추천합니다.
    """
    recommender = RecommendationService()
    
    try:
        # 추천 로직 실행
        recommendations = recommender.recommend_perfumes(description)
        return {"result": recommendations, "message": "추천 성공"}
    
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        # 예기치 못한 오류 처리
        raise HTTPException(status_code=500, detail=f"추천 중 오류 발생: {str(e)}")

