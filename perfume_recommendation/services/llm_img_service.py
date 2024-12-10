from fastapi import HTTPException
import logging
from models.img_llm_client import GPTClient

logger = logging.getLogger(__name__)

class LLMImageService:
    def __init__(self, gpt_client: GPTClient):
        self.gpt_client = gpt_client

    def generate_image_description(self, user_input: str) -> str:
        try:
            image_prompt = f"Please write a detailed description of the image based on the following keywords: {user_input} Please make sure that your answers are in English. You do not need to explain what the user's keywords are when you answer."
            imageGeneratePrompt = self.gpt_client.generate_response(image_prompt)  
            if not imageGeneratePrompt:
                raise ValueError("Failed to generate image description.")
            return imageGeneratePrompt
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate image description.")
