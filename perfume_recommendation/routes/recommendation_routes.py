from fastapi import APIRouter, HTTPException , Body
from services.recommendation_service import RecommendationService  
from services.img_recommendation_service import Img_RecommendationService
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, root_validator
from typing import Optional

# 응답 모델 정의
class RecommendationResponse(BaseModel):
    result: dict = {
        "recommendation": str,
        "image_url": str
    }

class ImageRecommendationRequest(BaseModel):
    user_input: Optional[str] = None
    image_url: Optional[str] = None

    @root_validator(pre=True)
    def check_at_least_one_field(cls, values):
        user_input, image_url = values.get("user_input"), values.get("image_url")
        
        # 둘 중 하나는 반드시 존재해야 함
        if not user_input and not image_url:
            raise ValueError("Either 'user_input' or 'image_url' must be provided.")
        
        # 둘 다 존재할 수 있음
        # 단, 둘 다 존재해도 에러가 발생하지 않도록 수정
        return values

router = APIRouter()

@router.post("/image", response_model=dict)
async def recommend_image(data: ImageRecommendationRequest = Body(...)):
    """
    텍스트 또는 이미지 URL을 기반으로 향수를 추천합니다.
    """
    try:
        user_input = data.user_input
        image_url = data.image_url

        # 둘 중 하나가 반드시 존재해야 하며, 둘 다 존재하는 경우에도 작동
        if not user_input and not image_url:
            raise HTTPException(status_code=400, detail="Either 'user_input' or 'image_url' must be provided.")

        img_recommendation_service = Img_RecommendationService()

        # 이미지 URL을 사용하는 경우
        if image_url:
            result = img_recommendation_service.img_recommend_perfumes(user_input=user_input, image_url=image_url)
        else:
            result = img_recommendation_service.img_recommend_perfumes(user_input=user_input)

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")