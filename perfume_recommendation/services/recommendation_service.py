import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional
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

    def recommend_perfumes(self, user_input: Optional[str] = None, image_data: Optional[bytes] = None) -> str:
        """
        사용자 입력(텍스트 및 이미지)을 기반으로 향수를 추천합니다.
        - user_input: 텍스트 설명
        - image_data: 이미지 파일 데이터
        """
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")
        
        # 이미지 데이터를 기반으로 텍스트 추출 (이미지가 제공된 경우)
        image_text = ""
        if image_data:
            image_text = self.gpt_client.process_image(image_data)  # 수정된 메서드 호출
            print(f"이미지에서 추출된 텍스트: {image_text}")
        
        # 사용자 입력과 이미지 텍스트를 결합
        combined_input = f"{user_input}\n이미지 분석 결과: {image_text}" if image_text else user_input

        if not combined_input:
            raise ValueError("사용자 입력 또는 이미지 데이터를 제공해야 합니다.")

        # 향수 데이터 텍스트 준비
        perfumes_text = "\n".join([
            f"{perfume['name']} ({perfume['brand']}): {perfume['description']}]"
            for perfume in perfumes
        ])
        
        # GPT를 이용해 향수를 추천
        prompt = self.gpt_client.create_prompt(combined_input, perfumes_text)
        gpt_response = self.gpt_client.get_response(prompt)
        
        return gpt_response
