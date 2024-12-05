from fastapi import APIRouter, Depends
from services.llm_img_service import LLMImageService
from models.img_llm_client import GPTClient
from services.prompt_loader import PromptLoader  
import os

template_path = os.path.join(os.path.dirname(__file__), "..", "models", "prompt_template.json")

prompt_loader = PromptLoader(template_path)

router = APIRouter()

def get_llm_image_service():
    gpt_client = GPTClient(prompt_loader) 
    llm_image_service = LLMImageService(gpt_client)
    return llm_image_service

@router.post("/generate-image-description")
async def generate_image_description(user_input: str, llm_image_service: LLMImageService = Depends(get_llm_image_service)):
    return {"imageGeneratePrompt": llm_image_service.generate_image_description(user_input)}
