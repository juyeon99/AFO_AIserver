from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from services.image_fetch_service import ImageFetchService
import logging
from fastapi.responses import Response
from PIL import Image

class ImageByteRequest(BaseModel):
    imagePath: str

router = APIRouter()
image_fetch_service = ImageFetchService()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/get-image")
async def get_image(request: ImageByteRequest):
    """
    Fetch and return the image as raw bytes.
    """
    try:
        image_path = request.imagePath
        return image_fetch_service.get_image(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

