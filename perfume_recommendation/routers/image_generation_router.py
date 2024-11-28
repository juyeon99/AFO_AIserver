from fastapi import APIRouter, HTTPException
from services.image_generation_service import ImageGenerationService
import logging

router = APIRouter()
image_generation_service = ImageGenerationService()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/generate-image")
async def generate_image(prompt: str):
    """
    텍스트 프롬프트를 기반으로 이미지를 생성합니다.
    """
    try:
        logger.info(f"Received prompt: {prompt}")
        output_path = image_generation_service.generate_image(prompt)
        logger.info(f"Generated image path: {output_path}")
        return {"message": "이미지 생성 성공", "output_path": output_path}
    except ValueError as e:
        logger.error(f"Error in image generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
