from fastapi import APIRouter, UploadFile, File, HTTPException
from services.image_service import ImageService

router = APIRouter()
service = ImageService()

@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        return service.process_image(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
