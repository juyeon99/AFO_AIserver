import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional
from models.llm_client import GPTClient
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
        self.gpt_client = GPTClient()

    def fetch_data_from_db(self) -> List[Dict]:
        # DB에서 데이터를 가져오는 메서드
        pass

    def recommend_perfumes(self, user_input: Optional[str] = None, image_data: Optional[bytes] = None) -> str:
        try:
            # 추천 로직 실행
            perfumes = self.fetch_data_from_db()
            if not perfumes:
                raise ValueError("향수 데이터를 가져오지 못했습니다.")
            
            # GPT를 통한 추천
            result = self.gpt_client.recommend(user_input, image_data, perfumes)
            return result
        except Exception as e:
            raise Exception(f"추천 중 오류 발생: {str(e)}")


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

    def recommend_perfumes(self, user_input: Optional[str] = None, image_data: Optional[bytes] = None) -> str:
        """
        사용자 입력(텍스트 및 이미지)을 기반으로 향수를 추천합니다.
        - user_input: 텍스트 설명
        - image_data: 이미지 파일 데이터
        """
        # 향수 데이터를 가져옵니다.
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
        
        # 향수 데이터 텍스트 준비
        perfumes_text = "\n".join([f"{perfume['name']} ({perfume['brand']}): {perfume['description']}" for perfume in perfumes])

        # 사용자 입력 또는 이미지 데이터가 없으면 예외를 발생시킴
        if not user_input and not image_data:
            raise ValueError("사용자 입력 또는 이미지 데이터를 제공해야 합니다.")

        # GPTClient를 사용해 향수를 추천
        try:
            # 실제 GPTClient 추천 로직을 호출합니다.
            return self.gpt_client.recommend(user_input, image_data, perfumes_text)
        except Exception as e:
            raise ValueError(f"향수 추천 중 오류 발생: {e}")
