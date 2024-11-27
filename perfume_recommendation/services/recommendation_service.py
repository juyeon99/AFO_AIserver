import mysql.connector
from mysql.connector import Error
from typing import List, Dict
from models.llm_client import GPTClient  # GPTClient 추가
from dotenv import load_dotenv
import os, uuid, logging

logger = logging.getLogger(__name__)

load_dotenv()

class RecommendationService:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME")
        }
        self.gpt_client = GPTClient()  # GPTClient 인스턴스 생성

    def fetch_data_from_db(self) -> List[Dict]:
        """데이터베이스에서 향수 데이터를 가져옵니다."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)

            # 향수 데이터를 조회
            query = """
            SELECT id, name, brand, description
            FROM perfume
            """
            cursor.execute(query)
            perfumes = cursor.fetchall()
            return perfumes

        except Error as e:
            print(f"데이터베이스 연결 오류: {e}")
            return []

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def translate_to_english(self, korean_text: str) -> str:
        """GPTClient를 통해 번역 수행"""
        return self.gpt_client.translate_to_english(korean_text)

    def create_image_prompt(self, user_input: str) -> str:
        # 한국어 입력을 영어로 번역
        english_input = self.gpt_client.translate_to_english(user_input)
        
        prompt = f"""
        Visual interpretation of {english_input} transformed into a suitable atmospheric background
        """
        return prompt

    def recommend_perfumes(self, user_input: str) -> dict:
        """
        사용자 입력에 맞는 향수를 추천하고 관련 이미지를 생성합니다.
        """
        try:
            # 향수 데이터 필터링
            filtered_perfumes = self.filter_recommendations(user_input)
            perfumes = filtered_perfumes if filtered_perfumes else self.fetch_data_from_db()
            
            if not perfumes:
                raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
            
            # GPT-4에 전달할 텍스트 생성
            perfumes_text = "\n".join([f"{perfume['name']}: {perfume['description']}" for perfume in perfumes])
            
            # GPT-4 호출하여 추천 결과 얻기
            gpt_response = self.gpt_client.get_response(user_input=user_input, perfumes_text=perfumes_text)
            
            # 이미지 프롬프트 작성(content moderation 문제 피하기 위해)
            image_prompt = self.create_image_prompt(user_input)
            
            # 사용자 입력을 직접 이미지 생성에 사용
            s3_key = f"perfume_images/{uuid.uuid4()}.png"
            image_url = self.gpt_client.generate_and_upload_image(image_prompt, s3_key)
            
            return {
                "recommendation": gpt_response,
                "image_url": image_url
            }
        
        except Exception as e:
            print(f"추천 생성 중 오류 발생: {str(e)}")
            raise

    def filter_recommendations(self, user_input: str) -> List[Dict]:
        """
        사용자 입력을 기반으로 향수를 필터링합니다.
        """
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
    
    # 필터링된 향수 목록 반환
        return [
            perfume for perfume in perfumes 
            if user_input.lower() in (perfume.get('description') or '').lower()
        ]