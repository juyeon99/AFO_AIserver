from fastapi import APIRouter, HTTPException, UploadFile, Form, Body
from pydantic import BaseModel
from services.recommendation_service import RecommendationService  
from services.img_recommendation_service import Img_RecommendationService

# 응답 모델 정의
class RecommendationResponse(BaseModel):
    result: dict = {
        "recommendation": str,
        "image_url": str
    }

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(user_input: str = Body(...)):
    try:
        recommendation_service = RecommendationService()
        result = recommendation_service.recommend_perfumes(user_input)
        
        # result는 이미 올바른 형식 (recommendation과 image_url을 포함)으로 반환되고 있으므로
        # 그대로 반환
        return {"result" : result}

    except Exception as e:
        # 에러 처리
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
            
@router.post("/image", response_model=dict)
async def recommend_image(user_input: str = Form(...), image: UploadFile = None):
    """
    텍스트와 이미지 데이터를 기반으로 향수를 추천합니다.
    """
    try:
        image_data = await image.read() if image else None
        img_recommendation_service = Img_RecommendationService()
        result = img_recommendation_service.img_recommend_perfumes(user_input=user_input, image_data=image_data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")