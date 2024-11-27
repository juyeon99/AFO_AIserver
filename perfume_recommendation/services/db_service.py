import mysql.connector
from mysql.connector import Error
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

class DBService:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }

    def fetch_perfume_data(self) -> List[Dict]:
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