import mysql.connector
from mysql.connector import Error
from typing import List, Dict
from dotenv import load_dotenv
import os
import json
import time

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
        self.cache_file = "perfume_cache.json"  # 캐시 파일 경로
        self.cache_timestamp_file = "cache_timestamp.txt"  # 캐시 타임스탬프 파일
        self.cache_duration = 2592000  # 캐시 만료 시간 (1시간)

    def load_perfume_data_from_db(self) -> List[Dict]:
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

    def is_cache_valid(self) -> bool:
        """
        캐시 파일의 타임스탬프를 확인하여 캐시가 유효한지 확인합니다.
        """
        if not os.path.exists(self.cache_timestamp_file):
            return False  # 캐시 타임스탬프 파일이 없으면 캐시가 유효하지 않음

        with open(self.cache_timestamp_file, "r") as f:
            last_timestamp = float(f.read())

        current_time = time.time()
        if current_time - last_timestamp < self.cache_duration:
            return True  # 캐시가 유효함

        return False  # 캐시가 만료됨

    def load_perfume_data(self) -> List[Dict]:
        """
        캐시 파일에서 향수 데이터를 로드합니다.
        캐시가 유효하지 않으면 데이터베이스에서 데이터를 로드하여 캐시를 갱신합니다.
        """
        if self.is_cache_valid():
            # 캐시가 유효하면 캐시된 데이터를 로드
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                print("캐시에서 향수 데이터 로드")
                return data
        else:
            # 캐시가 만료되었거나 없으면 데이터베이스에서 데이터 로드
            perfume_data = self.load_perfume_data_from_db()
            # 데이터베이스에서 가져온 데이터를 캐시 파일에 저장
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(perfume_data, f, ensure_ascii=False, indent=2)
            # 캐시 타임스탬프 갱신
            with open(self.cache_timestamp_file, "w") as timestamp_f:
                timestamp_f.write(str(time.time()))
            print("데이터베이스에서 향수 데이터를 로드하여 캐시 갱신")
            return perfume_data

    def fetch_perfume_data(self) -> List[Dict]:
        """
        향수 데이터를 반환합니다 (메모리에서 가져옵니다).
        """
        return self.load_perfume_data()
