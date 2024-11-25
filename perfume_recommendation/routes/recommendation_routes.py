from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from services.recommendation_service import RecommendationService

class RecommendationResponse(BaseModel):
    result: dict

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(user_input: str = "", image: UploadFile = None):
    try:
        if not user_input and not image:
            raise HTTPException(status_code=400, detail="user_input 또는 image가 필요합니다.")
        
        # image가 제공되면 파일 데이터를 읽어옴
        image_data = await image.read() if image else None

        # RecommendationService 인스턴스 생성
        recommendation_service = RecommendationService()
        
        # 추천 로직 실행
        result = recommendation_service.recommend_perfumes(user_input=user_input, image_data=image_data)
        
        # 결과 반환
        return {"result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
