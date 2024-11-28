from fastapi import APIRouter, HTTPException
from services.image_generation_service import ImageGenerationService

router = APIRouter()
image_generation_service = ImageGenerationService()

@router.post("/generate-image")
async def generate_image(prompt: str):
    """
    텍스트 프롬프트를 기반으로 이미지를 생성합니다.
    """
    try:
        output_path = image_generation_service.generate_image(prompt)
        return {"message": "이미지 생성 성공", "output_path": output_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
