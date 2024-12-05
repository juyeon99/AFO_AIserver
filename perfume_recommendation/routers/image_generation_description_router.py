from fastapi import APIRouter, Depends
from pydantic import BaseModel
from services.llm_img_service import LLMImageService
from models.img_llm_client import GPTClient
from services.prompt_loader import PromptLoader  
import os

# 요청 바디 모델 정의
class ImageDescriptionRequest(BaseModel):
    user_input: str

# 템플릿 경로 설정
template_path = os.path.join(os.path.dirname(__file__), "..", "models", "prompt_template.json")
prompt_loader = PromptLoader(template_path)

# FastAPI 라우터 설정
router = APIRouter()

def get_llm_image_service():
    gpt_client = GPTClient(prompt_loader)
    llm_image_service = LLMImageService(gpt_client)
    return llm_image_service

@router.post("/generate-image-description")
async def generate_image_description(
    request: ImageDescriptionRequest, 
    llm_image_service: LLMImageService = Depends(get_llm_image_service)
):
    return {"imageGeneratePrompt": llm_image_service.generate_image_description(request.user_input)}
