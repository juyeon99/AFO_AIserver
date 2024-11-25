import mysql.connector
from mysql.connector import Error
from typing import List, Dict
from models.llm_client import GPTClient  # GPTClient 추가
from dotenv import load_dotenv
import os

load_dotenv()

class RecommendationService:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST"),  # 기본값은 localhost
            "port": os.getenv("DB_PORT"),        # 기본값은 3306
            "user": os.getenv("DB_USER"),      # 기본값은 root
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

    def get_all_recommendations(self) -> List[Dict]:
        """모든 향수를 반환합니다."""
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
        return perfumes

    def filter_recommendations(self, user_input: str) -> List[Dict]:
        """사용자 입력을 기반으로 향수를 필터링합니다."""
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
        
        filtered_perfumes = [
            perfume for perfume in perfumes 
            if user_input.lower() in (perfume.get('description') or '').lower()
        ]
        return filtered_perfumes

    def recommend_perfumes(self, user_input: str) -> str:
        """사용자 입력에 맞는 향수를 추천합니다."""
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
        
        # 향수 데이터 텍스트를 준비합니다.
        perfumes_text = "\n".join([
            f"{perfume['name']}: {perfume['description']}" for perfume in perfumes
        ])
        
        # GPT를 이용해 향수를 추천합니다.
        prompt = self.gpt_client.create_prompt(user_input, perfumes_text)
        gpt_response = self.gpt_client.get_response(prompt)
        
        return gpt_response
