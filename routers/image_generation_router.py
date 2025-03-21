from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from services.image_generation_service import ImageGenerationService
import logging


class ImageRequest(BaseModel):
    imageGeneratePrompt: str


router = APIRouter()
image_generation_service = ImageGenerationService()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/generate-image")
async def generate_image(request: ImageRequest):
    """
    텍스트 프롬프트를 기반으로 이미지를 생성합니다.
    """
    try:
        logger.info(f"Received imageGeneratePrompt: {request.imageGeneratePrompt}")
        output_path = image_generation_service.generate_image(
            request.imageGeneratePrompt
        )
        logger.info(f"Generated image path: {output_path}")
        return {"path": output_path["absolute_path"]}
    except ValueError as e:
        logger.error(f"Error in image generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
