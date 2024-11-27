from models.img_llm_client import GPTClient

class ImageProcessingService:
    def __init__(self):
        self.gpt_client = GPTClient()

    def process_image(self, image_data: bytes) -> dict:
        description = self.gpt_client.process_image(image_data)
        feeling = self.gpt_client.generate_image_feeling(description)
        return {"description": description, "feeling": feeling}