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

    def recommend_perfumes(self, user_input: str) -> str:
        """
        사용자 입력에 맞는 향수를 추천합니다.
        """
    # 향수 데이터를 필터링
        filtered_perfumes = self.filter_recommendations(user_input)
    
    # 필터링된 결과가 없을 경우 전체 데이터를 사용
        perfumes = filtered_perfumes if filtered_perfumes else self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
    
    # GPT-4에 전달할 텍스트 생성
        perfumes_text = "\n".join([f"{perfume['name']}: {perfume['description']}" for perfume in perfumes])
    
        try:
        # GPT-4 호출
            gpt_response = self.gpt_client.get_response(user_input=user_input, perfumes_text=perfumes_text)
            return gpt_response
        except Exception as e:
            print(f"추천 생성 중 오류 발생: {str(e)}")
        return "추천 생성 중 오류가 발생했습니다."

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