from fastapi import HTTPException
import logging
from models.img_llm_client import GPTClient

logger = logging.getLogger(__name__)

class LLMImageService:
    def __init__(self, gpt_client: GPTClient):
        self.gpt_client = gpt_client

    def generate_image_description(self, user_input: str) -> str:
        try:
            image_prompt = f"""Describe the essence of the scene based on the following keywords: {user_input}. 
            Focus solely on the scents, atmosphere, and emotions evoked by the image. 
            Describe the way the air feels, the intensity of the fragrance, and how different notes blend together. 
            Capture the mood and emotional depth without referring to any physical objects or visual elements. 
            Use expressive and immersive language to create a sensory-rich experience. Your response should be in English. 
            Avoid mentioning any perfume bottles, containers, or tangible items—only describe the feeling and scent itself.
            """
            imageGeneratePrompt = self.gpt_client.generate_response(image_prompt)  
            if not imageGeneratePrompt:
                raise ValueError("Failed to generate image description.")
            return imageGeneratePrompt
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate image description.")
