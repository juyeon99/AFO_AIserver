from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.image_processing_service import ImageProcessingService

router = APIRouter()
image_processing_service = ImageProcessingService()

class ImageURLRequest(BaseModel):
    image_url: str

@router.post("/process-image")
async def process_image(request: ImageURLRequest):
    """
    이미지 URL을 처리하여 설명과 느낌을 반환합니다.
    """
    try:
        # 이미지 데이터 다운로드
        image_data = image_processing_service.download_image_from_url(request.image_url)
        # 이미지 처리
        result = image_processing_service.process_image(image_data)
        return {"message": "이미지 처리 성공", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
