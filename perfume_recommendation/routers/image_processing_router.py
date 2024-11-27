from fastapi import APIRouter, UploadFile
from models.img_llm_client import GPTClient

router = APIRouter()

@router.post("/process-image")
async def process_image(file: UploadFile):
    client = GPTClient()
    image_data = await file.read()
    description = client.process_image(image_data)
    feeling = client.generate_image_feeling(description)
    return {"description": description, "feeling": feeling}
