from fastapi import APIRouter
from models.img_llm_client import GPTClient
from datetime import datetime

router = APIRouter()

@router.post("/generate-image")
async def generate_image(prompt: str):
    client = GPTClient()
    output_filename = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"
    client.generate_image(prompt, output_filename)
    return {"message": "Image generation started", "output_filename": output_filename}