# from typing import Optional
# from models.img_llm_client import GPTClient
# from models.img_recommendation_service import Img_RecommendationService

# class LLMService:
#     def __init__(self):
#         self.gpt_client = GPTClient()
#         self.recommendation_service = Img_RecommendationService()

#     def recommend_perfume(self, user_input: str, image_url: Optional[str] = None) -> str:
#         perfumes = self.recommendation_service.fetch_data_from_db()
#         if not perfumes:
#             raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
        
#         perfumes_text = "\n".join(
#             [f"{perfume['name']}: {perfume['description']}" for perfume in perfumes]
#         )
        
#         image_description = ""
#         image_feeling = ""
#         if image_url:
#             try:
#                 image_data = self.recommendation_service.download_image_from_url(image_url)
#                 image_description = self.gpt_client.process_image(image_data)
#                 image_feeling = self.gpt_client.generate_image_feeling(image_description)
#             except ValueError:
#                 image_description = "이미지를 처리할 수 없습니다."
#                 image_feeling = "이미지에서 특별한 느낌을 분석하지 못했습니다."
        
#         prompt = f"{user_input}\n이미지 설명: {image_description}\n이미지 느낌: {image_feeling}".strip()
#         return self.gpt_client.get_response(prompt, perfumes_text)
