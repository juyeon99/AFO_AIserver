from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, root_validator
from typing import Optional
from services.recommendation_service import RecommendationService
from services.img_recommendation_service import Img_RecommendationService

# 응답 모델 정의
class RecommendationResponse(BaseModel):
    result: str

class ImageRecommendationRequest(BaseModel):
    user_input: Optional[str] = None
    image_url: Optional[str] = None

    @root_validator(pre=True)
    def check_one_field_present(cls, values):
        user_input, image_url = values.get("user_input"), values.get("image_url")
        if not user_input and not image_url:
            raise ValueError("Either 'user_input' or 'image_url' must be provided.")
        if user_input and image_url:
            raise ValueError("Only one of 'user_input' or 'image_url' should be provided.")
        return values

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(user_input: str = Body(...)):
    try:
        # 서비스 호출
        recommendation_service = RecommendationService()

        # GPTClient를 통해 추론된 결과 가져오기
        result = recommendation_service.recommend_perfumes(user_input)

        return {"result": result}

    except Exception as e:
        # 에러 처리
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/image", response_model=dict)
async def recommend_image(data: ImageRecommendationRequest):
    """
    텍스트 또는 이미지 URL을 기반으로 향수를 추천합니다.
    """
    try:
        user_input = data.user_input
        image_url = data.image_url

        # 둘 중 하나가 반드시 존재해야 하며, 둘 다 존재하면 에러 처리
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
