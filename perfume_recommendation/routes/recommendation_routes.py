from fastapi import APIRouter, HTTPException, UploadFile, Form, Body
from pydantic import BaseModel
from services.recommendation_service import RecommendationService  
from services.img_recommendation_service import RecommendationService

# 응답 모델 정의
class RecommendationResponse(BaseModel):
    result: str

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(user_input: str = Body(...)):
    try:
        # RecommendationService 인스턴스를 사용하여 텍스트 입력에 대한 향수 추천 요청
        recommendation_service = RecommendationService()
        # perfumes_text는 예시로 제공되어야 하므로 실제 데이터에 맞게 전달
        perfumes_text = "향수 목록"  # 여기에 실제 향수 목록 데이터를 넣으세요
        result = recommendation_service.recommend_perfumes(user_input=user_input, perfumes_text=perfumes_text)

        return {"result": result}

    except Exception as e:
        # 예외 발생 시 500 상태 코드 반환
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")\
            
@router.post("/image", response_model=dict)
async def recommend_image(user_input: str = Form(...), image: UploadFile = None):
    """
    텍스트와 이미지 데이터를 기반으로 향수를 추천합니다.
    """
    try:
        image_data = await image.read() if image else None
        img_recommendation_service = RecommendationService()
        result = img_recommendation_service.recommend_perfumes(user_input=user_input, image_data=image_data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")