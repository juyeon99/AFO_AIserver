from fastapi import APIRouter, HTTPException, File, UploadFile, Body
from typing import Optional
from services.recommendation_service import RecommendationService

router = APIRouter()
service = RecommendationService()

# 사용자 입력(텍스트 또는 이미지)을 받아 향수를 추천하는 API
@router.post("/recommend")
async def recommend_perfume(
    description: Optional[str] = Body(None, description="향수에 대한 텍스트 설명"),
    image: Optional[UploadFile] = File(None, description="향수 관련 이미지 파일")
):
    """
    사용자 입력(description 또는 image)을 기반으로 향수를 추천합니다.
    둘 중 하나만 입력해도 동작합니다.
    """
    if not description and not image:
        raise HTTPException(
            status_code=400,
            detail="텍스트 설명(description) 또는 이미지(image) 중 하나를 제공해야 합니다."
        )
    
    recommender = RecommendationService()
    image_data = None
    
    try:
        # 이미지 파일이 제공되었으면 읽기
        if image:
            image_data = await image.read()
        
        # 추천 로직 실행
        recommendations = recommender.recommend_perfumes(description, image_data)
        return {"result": recommendations, "message": "추천 성공"}
    
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        # 예기치 못한 오류 처리
        raise HTTPException(status_code=500, detail=f"추천 중 오류 발생: {str(e)}")
