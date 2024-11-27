import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional
from models.img_llm_client import GPTClient
from dotenv import load_dotenv
from datetime import datetime
import os, requests

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

    def download_image_from_url(self, image_url: str) -> bytes:
        """
        이미지 URL에서 이미지를 다운로드하여 바이트 데이터로 반환합니다.
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return response.content  # 이미지 데이터를 바이트 형태로 반환
        except requests.RequestException as e:
            raise ValueError(f"이미지 URL에서 데이터를 가져오는 데 실패했습니다: {e}")

    def img_recommend_perfumes(self, user_input: str, image_url: Optional[str] = None) -> str:
        """
        텍스트와 이미지 URL 데이터를 기반으로 향수를 추천하고 이미지를 생성하여 저장합니다.
        """
        # 데이터베이스에서 향수 데이터 가져오기
        perfumes = self.fetch_data_from_db()
        if not perfumes:
            raise ValueError("데이터베이스에서 향수 데이터를 가져오지 못했습니다.")

    # 향수 데이터를 텍스트로 구성
        perfumes_text = "\n".join(
        [f"{perfume['name']}: {perfume['description']}" for perfume in perfumes]
        )

    # 이미지 URL 처리
        image_description = ""
        image_feeling = ""
        if image_url:
            try:
                # 이미지 다운로드 후 처리
                image_data = self.download_image_from_url(image_url)
                image_description = self.gpt_client.process_image(image_data)
                image_feeling = self.gpt_client.generate_image_feeling(image_description)
            except ValueError:
                # URL 처리 실패 시 기본 메시지 설정
                image_description = "이미지를 처리할 수 없습니다."
                image_feeling = "이미지에서 특별한 느낌을 분석하지 못했습니다."

    # 사용자 입력과 이미지 정보를 결합
        prompt = f"{user_input}\n이미지 설명: {image_description}\n이미지 느낌: {image_feeling}".strip()

    # GPT-4를 호출하여 추천 생성
        recommendation_response = self.gpt_client.get_response(prompt, perfumes_text)

    # 이미지를 생성하고 저장
        if image_description:
            output_filename = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"  # 자동 파일 이름 생성
            self.gpt_client.generate_image(prompt, output_filename)
    
        return recommendation_response
