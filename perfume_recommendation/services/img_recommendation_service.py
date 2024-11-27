import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional
from models.img_llm_client import GPTClient
from dotenv import load_dotenv
import os

load_dotenv()

class Img_RecommendationService:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
        self.gpt_client = GPTClient()  # GPTClient 인스턴스 생성

    def fetch_data_from_db(self) -> List[Dict]:
        """
        데이터베이스에서 향수 데이터를 가져옵니다.
        """
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)

            query = "SELECT id, name, brand, description FROM perfume"
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

    def recommend_perfumes(self, user_input: str, image_data: Optional[bytes] = None) -> str:
        """
        텍스트와 이미지 데이터를 기반으로 향수를 추천합니다.
        """
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")

        # 향수 데이터를 텍스트로 구성
        perfumes_text = "\n".join(
            [f"{perfume['name']}: {perfume['description']}" for perfume in perfumes]
        )

        # 이미지 데이터를 처리하여 설명과 느낌을 생성
        image_description = ""
        image_feeling = ""
        if image_data:
            image_description = self.gpt_client.process_image(image_data)
            image_feeling = self.gpt_client.generate_image_feeling(image_description)

        # 사용자 입력과 이미지 정보를 결합
        combined_input = f"{user_input}\n이미지 설명: {image_description}\n이미지 느낌: {image_feeling}".strip()

        # GPT-4를 호출하여 추천 생성
        return self.gpt_client.get_response(combined_input, perfumes_text)
