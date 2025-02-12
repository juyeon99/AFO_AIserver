import logging
import json
import pymysql
import random
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from perfume_recommendation.models.base_model import Base, Product, Note, Spice, ProductImage, Similar, SimilarText, SimilarImage

logger = logging.getLogger(__name__)

# SQLAlchemy 설정
DATABASE_URL = "mysql+pymysql://banghyang:banghyang@192.168.0.182:3306/banghyang"
engine = create_engine(DATABASE_URL, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DBService:
    def __init__(
        self, db_config: Dict[str, str], cache_path: str = "perfume_cache.json"
    ):
        self.db_config = db_config
        self.connection = self.connect_to_db()
        self.cache_path = Path(cache_path)
        self.cache_expiration = timedelta(days=1)  # 캐싱 만료 시간 (1일)
        self.session = SessionLocal()

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()

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

    def fetch_brands(self) -> List[str]:
        """DB에서 브랜드 목록을 가져옵니다."""
        query = "SELECT DISTINCT brand FROM product;"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                brands = [row["brand"] for row in cursor.fetchall()]
            
            logger.info(f"✅ 총 {len(brands)}개의 브랜드 조회 완료")
            return brands
        except pymysql.MySQLError as e:
            logger.error(f"🚨 브랜드 데이터 로드 실패: {e}")
            return []
    
    def fetch_spices_by_line(self, line_id: int) -> List[Dict]:
        """특정 계열(line_id)에 속하는 향료(spice) 목록 조회"""
        try:
            query = """
                SELECT id, name_kr 
                FROM spice 
                WHERE line_id = %s;
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query, (line_id,))
                spices = cursor.fetchall()
            
            if not spices:
                logger.warning(f"⚠️ 해당 계열 ID({line_id})에 속하는 향료가 없습니다.")
                return []

            logger.info(f"✅ 계열 ID({line_id})에 해당하는 향료 {len(spices)}개 조회 완료")
            return spices

        except pymysql.MySQLError as e:
            logger.error(f"🚨 향료 데이터 로드 실패: {e}")
            return []

    def fetch_line_data(self) -> List[Dict]:
        """
        line 테이블의 모든 데이터를 조회하여 반환.

        Returns:
            List[Dict]: line 테이블의 데이터를 포함한 리스트
        """
        query = "SELECT * FROM line;"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                lines = cursor.fetchall()

            logger.info(f"✅ line 테이블 데이터 {len(lines)}개 조회 완료")
            return lines
        except pymysql.MySQLError as e:
            logger.error(f"🚨 데이터베이스 오류 발생: {e}")
            return []
    
    def get_perfumes_by_middel_notes(self, spice_ids: List[int]) -> List[Dict]:
        """MIDDLE 타입의 노트를 포함한 향수를 검색"""
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
                WHERE p.category_id = 1
                AND n.spice_id IN ({spice_ids_str})
                AND n.note_type = 'MIDDLE'
                GROUP BY p.id, p.brand, p.name_kr, p.size_option
                ORDER BY matching_count DESC;
            """

            with self.connection.cursor() as cursor:
                cursor.execute(query)
                perfumes = cursor.fetchall()
                logger.info(f"✅ 전체 매칭되는 향수 {len(perfumes)}개를 찾았습니다.")

                return perfumes

        except pymysql.MySQLError as e:
            logger.error(f"🚨 향수 데이터 로드 실패: {e}")
            raise
    
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
        """향료 이름으로 ID를 가져옵니다."""
        try:
            # LIKE 검색을 위한 패턴 생성
            patterns = [f"name_kr LIKE '%{note.strip()}%'" for note in note_names] # 한글 이름으로 검색
            where_clause = " OR ".join(patterns) # OR 조건으로 연결
            
            query = f"""
                SELECT id, name_kr
                FROM spice 
                WHERE {where_clause}
                ORDER BY 
                    CASE 
                        WHEN name_kr IN ({', '.join([f"'{note.strip()}'" for note in note_names])}) THEN 0 
                        ELSE 1 
                    END,
                    name_kr;
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query) # 쿼리 실행
                result = cursor.fetchall() # 결과를 리스트로 반환
                
                logger.info(f"✅ 요청된 향료: {note_names}")
                logger.info(f"✅ 매칭된 향료: {[r['name_kr'] for r in result]}")
                
                return result
                
        except pymysql.MySQLError as e:
            logger.error(f"🚨 향료 데이터 로드 실패: {e}")
            raise

    def get_diffusers_by_spice_ids(self, spice_ids: List[int]) -> List[Dict]:
        """해당 향료가 하나라도 포함된 디퓨저들 중에서 랜덤하게 2개를 선택합니다."""
        try:
            spice_ids_str = ",".join(map(str, spice_ids))
            
            # 먼저 전체 매칭되는 디퓨저 수를 확인
            count_query = f"""
                SELECT COUNT(DISTINCT p.id) as total_count
                FROM product p
                JOIN note n ON p.id = n.product_id
                WHERE p.category_id = 2
                AND n.spice_id IN ({spice_ids_str})
                AND p.name_kr NOT LIKE '%카 디퓨저%'
            """
            
            # 그 다음 랜덤하게 2개 선택
            main_query = f"""
                SELECT DISTINCT
                    p.id, 
                    p.brand, 
                    p.name_kr, 
                    p.size_option as volume,
                    p.content,
                    COUNT(DISTINCT n.spice_id) as matching_count,
                    GROUP_CONCAT(DISTINCT s.name_kr) as included_notes
                FROM product p
                JOIN note n ON p.id = n.product_id
                JOIN spice s ON n.spice_id = s.id
                WHERE p.category_id = 2
                AND n.spice_id IN ({spice_ids_str})
                AND p.name_kr NOT LIKE '%카 디퓨저%'
                GROUP BY p.id, p.brand, p.name_kr, p.size_option, p.content
                ORDER BY RAND()
                LIMIT 2
            """
            
            with self.connection.cursor() as cursor:
                # 전체 개수 확인
                cursor.execute(count_query)
                total_count = cursor.fetchone()['total_count']
                logger.info(f"✅ 전체 매칭되는 디퓨저: {total_count}개")
                
                # 랜덤 선택
                cursor.execute(main_query)
                result = cursor.fetchall()
                
                # 선택된 디퓨저 로깅
                for diffuser in result:
                    logger.info(
                        f"✅ 선택됨: {diffuser['name_kr']} (ID: {diffuser['id']}) - "
                        f"포함 향료: {diffuser['included_notes']}"
                    )
                
                return result
                
        except pymysql.MySQLError as e:
            logger.error(f"🚨 디퓨저 데이터 로드 실패: {e}")
            raise
        
    # ORM을 사용하는 새로운 메서드들
    def get_product_by_id(self, product_id: int):
        """SQLAlchemy를 사용하여 제품 정보를 조회합니다."""
        try:
            return self.session.query(Product).filter(Product.id == product_id).first()
        except Exception as e:
            logger.error(f"🚨 제품 조회 실패: {e}")
            return None

    def get_similar_products_by_text(self, product_id: int) -> List[Dict]:
        """텍스트 기반 유사도로 비슷한 제품을 조회합니다."""
        try:
            similar_products = (
                self.session.query(
                    Product.id,
                    Product.brand,
                    Product.name_kr,
                    Product.size_option.label('volume'),
                    SimilarText.similarity_score
                )
                .join(SimilarText, Product.id == SimilarText.similar_product_id)
                .filter(SimilarText.product_id == product_id)
                .order_by(SimilarText.similarity_score.desc())
                .limit(5)
                .all()
            )
            logger.info(f"✅ 텍스트 기반 유사 제품 {len(similar_products)}개 조회 완료")
            return [dict(zip(['id', 'brand', 'name_kr', 'volume', 'similarity_score'], p)) for p in similar_products]
        except Exception as e:
            logger.error(f"🚨 텍스트 기반 유사 제품 조회 실패: {e}")
            return []


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
