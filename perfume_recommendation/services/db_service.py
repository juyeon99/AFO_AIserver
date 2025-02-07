import logging
import json
import pymysql
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DBService:
    def __init__(
        self, db_config: Dict[str, str], cache_path: str = "perfume_cache.json"
    ):
        self.db_config = db_config
        self.connection = self.connect_to_db()
        self.cache_path = Path(cache_path)
        self.cache_expiration = timedelta(days=1)  # 캐싱 만료 시간 (1일)

    def connect_to_db(self):
        try:
            connection = pymysql.connect(
                host=self.db_config["host"],
                port=int(self.db_config["port"]),
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"],
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
            logger.info("✅ 데이터베이스 연결 성공!")
            return connection
        except pymysql.MySQLError as e:
            logger.error(f"🚨 데이터베이스 연결 오류: {e}")
            return None

    # def fetch_line_data(self) -> List[Dict]:
    #     """
    #     line 테이블의 모든 데이터를 조회하여 반환.

    #     Returns:
    #         List[Dict]: line 테이블의 데이터를 포함한 리스트
    #     """
    #     query = "SELECT * FROM line;"
    #     try:
    #         with self.connection.cursor() as cursor:
    #             cursor.execute(query)
    #             lines = cursor.fetchall()

    #         logger.info(f"✅ line 테이블 데이터 {len(lines)}개 조회 완료")
    #         return lines
    #     except pymysql.MySQLError as e:
    #         logger.error(f"🚨 데이터베이스 오류 발생: {e}")
    #         return []

    def cache_perfume_data(self, force: bool = False) -> None:
        """
        DB의 향수 데이터를 JSON 파일로 캐싱. `force=True` 또는 변경 사항이 있을 경우 갱신.
        """
        existing_products = self.load_cached_perfume_data(check_only=True)

        query = """
        SELECT 
            p.id, p.name_kr, p.name_en, p.brand, p.main_accord, p.category_id
        FROM product p
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                new_products = cursor.fetchall()

            # 데이터 변경 여부 확인
            if not force and self.is_cache_up_to_date(existing_products, new_products):
                logger.info(f"✅ 캐싱 데이터가 최신 상태입니다: {self.cache_path}")
                return

            # 캐싱 파일 저장
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(new_products, f, ensure_ascii=False, indent=4)

            logger.info(f"✅ 향수 데이터를 JSON으로 캐싱 완료: {self.cache_path}")

        except pymysql.MySQLError as e:
            logger.error(f"🚨 데이터베이스 오류 발생: {e}")

    def load_cached_perfume_data(self, check_only: bool = False) -> List[Dict]:
        """
        캐싱된 데이터를 로드. 캐싱 파일이 없으면 check_only=False일 때 새로 생성.
        """
        if not self.cache_path.exists():
            if check_only:
                return []
            logger.info("캐싱 파일이 존재하지 않아 새로 생성합니다.")
            self.cache_perfume_data()

        with open(self.cache_path, "r", encoding="utf-8") as f:
            products = json.load(f)

        logger.info(f"✅ 캐싱된 향수 데이터 {len(products)}개 로드")
        return products

    def is_cache_up_to_date(self, existing_products: List[Dict], new_products: List[Dict]) -> bool:
        """
        기존 캐싱 데이터와 새로 가져온 DB 데이터를 비교하여 변경 사항이 있는지 확인.
        """
        existing_dict = {item['id']: item for item in existing_products}
        new_dict = {item['id']: item for item in new_products}

        # 새로운 ID가 추가되었거나 기존 데이터가 변경되었는지 확인
        if set(existing_dict.keys()) != set(new_dict.keys()):
            logger.info("🔄 새로운 향수 데이터가 추가됨. 캐싱을 갱신합니다.")
            return False

        for key in new_dict.keys():
            if existing_dict[key] != new_dict[key]:  # 데이터 변경 확인
                logger.info("🔄 기존 향수 데이터가 변경됨. 캐싱을 갱신합니다.")
                return False

        return True

    def force_generate_cache(self) -> None:
        """
        강제로 JSON 캐싱 파일을 생성하는 메서드.
        """
        logger.info("강제 캐싱 생성 요청을 받았습니다.")
        self.cache_perfume_data(force=True)
        logger.info("✅ 강제 캐싱 생성 완료.")


    def get_spices_by_names(self, note_names: List[str]) -> List[Dict]:
        """미리 정의된 향료들의 ID를 가져옵니다."""
        try:
            notes_str = ", ".join([f"'{note}'" for note in note_names])
            query = f"""
                SELECT id, name_kr 
                FROM spice 
                WHERE name_kr IN ({notes_str});
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                logger.info(f"✅ 카테고리 향료: {note_names}")
                logger.info(f"✅ 찾은 향료 정보: {result}")
                return result
                
        except pymysql.MySQLError as e:
            logger.error(f"🚨 향료 데이터 로드 실패: {e}")
            raise

    def get_diffusers_by_spice_ids(self, spice_ids: List[int]) -> List[Dict]:
        """해당 향료들이 포함된 디퓨저를 찾습니다."""
        try:
            spice_ids_str = ",".join(map(str, spice_ids))
            query = f"""
                SELECT DISTINCT
                    p.id, 
                    p.brand, 
                    p.name_kr, 
                    p.size_option as volume,
                    COUNT(DISTINCT n.spice_id) as matching_count
                FROM product p
                JOIN note n ON p.id = n.product_id
                WHERE p.category_id = 2
                AND n.spice_id IN ({spice_ids_str})
                AND p.name_kr NOT LIKE '%카 디퓨저%'
                GROUP BY p.id, p.brand, p.name_kr, p.size_option
                ORDER BY matching_count DESC;
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                all_diffusers = cursor.fetchall()
                logger.info(f"✅ 전체 매칭되는 디퓨저 {len(all_diffusers)}개를 찾았습니다.")
                
                if len(all_diffusers) > 2:
                    # 랜덤하게 2개 선택
                    import random
                    result = random.sample(all_diffusers, 2)
                else:
                    result = all_diffusers
                    
                logger.info(f"✅ 선택된 디퓨저: {[d['name_kr'] for d in result]}")
                return result
                
        except pymysql.MySQLError as e:
            logger.error(f"🚨 디퓨저 데이터 로드 실패: {e}")
            raise


# 캐싱 생성 기능 실행
if __name__ == "__main__":
    import os

    # DB 설정
    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
    }

    # DB 서비스 초기화
    db_service = DBService(db_config=db_config)

    # 강제 캐싱 생성 실행
    db_service.force_generate_cache()
    print("향수 데이터 강제 캐싱 완료!")
