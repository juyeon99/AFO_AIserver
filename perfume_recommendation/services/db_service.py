import logging
import json, os
import pymysql
import random
from openai import OpenAI
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
from langchain_openai import ChatOpenAI
from perfume_recommendation.models.base_model import Base, Product, Note, Spice, ProductImage, Similar, SimilarText, SimilarImage

logger = logging.getLogger(__name__)

database_url = os.getenv("DATABASE_URL")
pool_recycle_prot = int(os.getenv("POOL_RECYCLE"))

# SQLAlchemy 설정
DATABASE_URL = database_url
engine = create_engine(DATABASE_URL, pool_recycle=pool_recycle_prot)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DBService:
    def __init__(
        self, db_config: Dict[str, str], cache_path_prefix: str = "cache"
    ):
        self.db_config = db_config
        self.connection = self.connect_to_db()
        self.cache_path_prefix = Path(cache_path_prefix)
        self.cache_path_prefix.mkdir(exist_ok=True)
        self.cache_expiration = timedelta(days=1)  # 캐싱 만료 시간 (1일)
        self.session = SessionLocal()
        self.gpt_client = self.initialize_gpt_client()

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

    def initialize_gpt_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_HOST")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
    
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
    
    def get_perfumes_by_middle_notes(self, spice_ids: List[int]) -> List[Dict]:
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
    
    # def cache_perfume_data(self, force: bool = False) -> None:
    #     """
    #     DB의 향수 데이터를 JSON 파일로 캐싱. `force=True` 또는 변경 사항이 있을 경우 갱신.
    #     """
    #     existing_products = self.load_cached_perfume_data(check_only=True)

    #     query = """
    #     SELECT 
    #         p.id, p.name_kr, p.name_en, p.brand, p.main_accord, p.category_id
    #     FROM product p
    #     """
    #     try:
    #         with self.connection.cursor() as cursor:
    #             cursor.execute(query)
    #             new_products = cursor.fetchall()

    #         # 데이터 변경 여부 확인
    #         if not force and self.is_cache_up_to_date(existing_products, new_products):
    #             logger.info(f"✅ 캐싱 데이터가 최신 상태입니다: {self.cache_path}")
    #             return

    #         # 캐싱 파일 저장
    #         with open(self.cache_path, "w", encoding="utf-8") as f:
    #             json.dump(new_products, f, ensure_ascii=False, indent=4)

    #         logger.info(f"✅ 향수 데이터를 JSON으로 캐싱 완료: {self.cache_path}")

    #     except pymysql.MySQLError as e:
    #         logger.error(f"🚨 데이터베이스 오류 발생: {e}")

    def cache_data(self, query: str, cache_file: Path, key_field: str, force: bool = False) -> None:
        """
        DB 데이터를 JSON 파일로 캐싱. `force=True` 또는 변경 사항이 있을 경우 갱신.
        """
        existing_data = self.load_cached_data(cache_file, check_only=True)

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                new_data = cursor.fetchall()

            # 데이터 변경 여부 확인
            if not force and self.is_cache_up_to_date(existing_data, new_data):
                logger.info(f"✅ 캐싱 데이터가 최신 상태입니다: {cache_file}")
                return

            # 캐싱 파일 저장
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

            logger.info(f"✅ 데이터 캐싱 완료: {cache_file}")

        except pymysql.MySQLError as e:
            logger.error(f"🚨 데이터베이스 오류 발생: {e}")

    # def load_cached_perfume_data(self, check_only: bool = False) -> List[Dict]:
    #     """
    #     캐싱된 데이터를 로드. 캐싱 파일이 없으면 check_only=False일 때 새로 생성.
    #     """
    #     if not self.cache_path.exists():
    #         if check_only:
    #             return []
    #         logger.info("캐싱 파일이 존재하지 않아 새로 생성합니다.")
    #         self.cache_perfume_data()

    #     with open(self.cache_path, "r", encoding="utf-8") as f:
    #         products = json.load(f)

    #     logger.info(f"✅ 캐싱된 향수 데이터 {len(products)}개 로드")
    #     return products
    
    def load_cached_data(self, cache_file: Path, check_only: bool = False) -> List[Dict]:
        """
        캐싱된 데이터를 로드. 캐싱 파일이 없으면 check_only=False일 때 새로 생성.
        """
        if not cache_file.exists():
            if check_only:
                return []
            logger.info(f"캐싱 파일 {cache_file}이(가) 존재하지 않아 새로 생성합니다.")
            if "perfume_cache" in str(cache_file):
                self.cache_perfume_data()
            elif "diffuser_cache" in str(cache_file):
                self.cache_diffuser_data()
            elif "spice_cache" in str(cache_file):
                self.cache_spice_data()
            elif "note_cache" in str(cache_file):
                self.cache_note_data()

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"✅ 캐싱된 데이터 {len(data)}개 로드: {cache_file}")
        return data

    def is_cache_up_to_date(self, existing_products: List[Dict], new_products: List[Dict]) -> bool:
        """
        기존 캐싱 데이터와 새로 가져온 DB 데이터를 비교하여 변경 사항이 있는지 확인.
        """
        existing_dict = {item['id']: item for item in existing_products}
        new_dict = {item['id']: item for item in new_products}

        # 새로운 ID가 추가되었거나 기존 데이터가 변경되었는지 확인
        if set(existing_dict.keys()) != set(new_dict.keys()):
            logger.info("🔄 새로운 데이터가 추가됨. 캐싱을 갱신합니다.")
            return False

        for key in new_dict.keys():
            if existing_dict[key] != new_dict[key]:  # 데이터 변경 확인
                logger.info("🔄 기존 데이터가 변경됨. 캐싱을 갱신합니다.")
                return False

        return True

    def force_generate_cache(self) -> None:
        """
        강제로 JSON 캐싱 파일을 생성하는 메서드.
        """
        logger.info("강제 캐싱 생성 요청을 받았습니다.")
        # self.cache_perfume_data(force=True)
        self.cache_perfume_data()
        self.cache_diffuser_data()
        self.cache_note_data()
        self.cache_spice_data()
        logger.info("✅ 강제 캐싱 생성 완료.")

    def cache_note_data(self) -> None:
        query = """
        SELECT id, note_type, product_id, spice_id FROM note
        """
        self.cache_data(query, self.cache_path_prefix / "note_cache.json", key_field="id")
    
    def cache_perfume_data(self) -> None:
        query = """
        SELECT p.id, p.name_kr, p.name_en, p.brand, p.main_accord, p.category_id FROM product p WHERE p.category_id = 1
        """
        self.cache_data(query, self.cache_path_prefix / "perfume_cache.json", key_field="id")

    def cache_diffuser_data(self) -> None:
        query = """
        SELECT p.id, p.name_kr, p.name_en, p.brand, p.category_id, p.content FROM product p WHERE p.category_id = 2
        """
        self.cache_data(query, self.cache_path_prefix / "diffuser_cache.json", key_field="id")

    def cache_spice_data(self) -> None:
        query = """
        SELECT id, content_en, content_kr, name_en, name_kr, line_id FROM spice
        """
        self.cache_data(query, self.cache_path_prefix / "spice_cache.json", key_field="id")
    
    def load_cached_note_data(self) -> List[Dict]:
        """
        Load cached note data from note_cache.json.
        """
        return self.load_cached_data(self.cache_path_prefix / "note_cache.json")
    
    def load_cached_perfume_data(self) -> List[Dict]:
        """
        Load cached spice data from perfume_cache.json.
        """
        return self.load_cached_data(self.cache_path_prefix / "perfume_cache.json")
    
    def load_cached_diffuser_data(self) -> List[Dict]:
        """
        Load cached spice data from perfume_cache.json.
        """
        return self.load_cached_data(self.cache_path_prefix / "diffuser_cache.json")

    def load_cached_spice_data(self) -> List[Dict]:
        """
        Load cached spice data from spice_cache.json.
        """
        return self.load_cached_data(self.cache_path_prefix / "spice_cache.json")
    
    def get_product_details(self, product_id, products):
        for product in products:
            if product["id"] == product_id:
                return product
        return None
    
    def generate_scent_description(self, notes_text, diffuser_description):
        prompt = f"""Based on the following fragrance combination of the diffuser, describe the characteristics of the overall scent using common perfumery terms such as 우디, 플로럴, 스파이시, 시트러스, 허브, 머스크, 아쿠아, 그린, 구르망, 푸제르, 알데하이드, 파우더리, 스모키, 프루티, 오리엔탈, etc. You do not need to break down each note, just focus on the overall scent impression.
            # EXAMPLE 1:
            - Note: Top: 이탈리안 레몬 잎, 로즈마리\nMiddle: 자스민, 라반딘\nBase: 시더우드, 머스크
            - Diffuser Description: 당신의 여정에 감각적이고 신선한 향기가 퍼집니다. 아침 햇살이 창문을 통해 들어올 때, 산들 바람과 함께 이탈리아 시골을 연상시키는 푸른 향기
            - Response: 상쾌한 시트러스와 허브의 조화, 플로럴한 우아함, 따뜻한 우디한 향과 부드러운 머스크가 어우러져 균형 잡힌 향기를 만들어냅니다. 전체적으로 이 향은 활력을 주면서도 동시에 편안함과 안정감을 느낄 수 있는, 다채롭고 매력적인 향입니다.

            # EXAMPLE 2:
            - Note: single: 이탈리안 베르가못, 이탈리안 레몬, 자몽, 무화과, 핑크 페퍼, 자스민 꽃잎, 무화과 나무, 시더우드, 벤조인
            - Diffuser Description: 당신의 여정에 감각적이고 신선한 향기가 퍼집니다. 아침 햇살이 창문을 통해 들어올 때, 산들 바람과 함께 이탈리아 시골을 연상시키는 푸른 향기
            - Response: 이 향은 상쾌하고 활기찬 느낌을 주면서도, 부드럽고 따뜻한 깊이를 지닌 균형 잡힌 향입니다. 밝고 톡톡 튀는 시트러스 향이 기분을 상쾌하게 해주고, 달콤하고 우아한 플로럴과 자연적인 우디한 느낌이 조화를 이루며 세련된 분위기를 만들어냅니다. 전체적으로 신선하고 세련되며, 따뜻하면서도 편안한 느낌을 주는 복합적인 향입니다.

            # Note: {notes_text}
            # Diffuser Description: {diffuser_description}
            # Response: """
        
        response = self.gpt_client.invoke(prompt).content.strip()

        return response

    # Load or initialize the diffuser scent cache
    def load_diffuser_scent_cache(self):
        """Load diffuser scent descriptions."""
        try:
            with open(self.cache_path_prefix / "diffuser_scent_cache.json", "r", encoding="utf-8") as f:
                return {item["id"]: item["scent_description"] for item in json.load(f)}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading diffuser scent data: {e}")
            return {}
    
    def format_notes(self, note_data):
        if "SINGLE" in note_data:
            single_notes = ", ".join(note_data["SINGLE"])
            return f"single: {single_notes}"
        else:
            formatted = []
            for note_type in ["TOP", "MIDDLE", "BASE"]:
                if note_data.get(note_type):
                    notes_str = ", ".join(note_data[note_type])
                    formatted.append(f"{note_type.lower()}: {notes_str}")
            return "\n".join(formatted)

    def save_scent_cache(self, scent_cache):
        # Update scent cache to a list before saving
        scent_cache_list = [{"id": int(product_id), "scent_description": scent_description} 
                            for product_id, scent_description in scent_cache.items()]
        with open("cache/diffuser_scent_cache.json", "w", encoding="utf-8") as f:
            json.dump(scent_cache_list, f, ensure_ascii=False, indent=4)

    def save_diffuser_scent_description(self) -> None:
        notes = self.load_cached_note_data()
        spices = self.load_cached_spice_data()
        products = self.load_cached_diffuser_data()

        # Extract product IDs from the product cache
        existing_product_ids = {product["id"] for product in products}

        # Create spice ID to name mapping
        spice_id_to_name = {spice["id"]: spice["name_kr"] for spice in spices}

        # Group notes by product_id
        product_notes = defaultdict(lambda: defaultdict(list))

        note_types = ["TOP", "MIDDLE", "BASE", "SINGLE"]

        for note in notes:
            note_type = note["note_type"].upper()
            product_id = note["product_id"]
            if product_id in existing_product_ids:
                spice_name = spice_id_to_name.get(note["spice_id"], "")
                if note_type in note_types and spice_name:
                    product_notes[product_id][note_type].append(spice_name)
        
        # Load the scent cache as a dictionary
        scent_cache = self.load_diffuser_scent_cache()

        # Generate and cache scent descriptions
        scent_cache_list = []

        for product_id, note_data in product_notes.items():
            if str(product_id) in scent_cache:
                logger.info(f"Product {product_id} already has a cached scent description.")
                scent_cache_list.append({
                    "id": int(product_id),
                    "scent_description": scent_cache[str(product_id)]
                })
                continue

            formatted_notes = self.format_notes(note_data)
            logger.info(f"Generating scent description for product {product_id}...")

            product_details = self.get_product_details(product_id, products)
            if product_details:
                # Diffuser description is fetched from product details or assigned manually
                diffuser_description = product_details.get("content", "")

            scent_description = self.generate_scent_description(formatted_notes, diffuser_description)
            scent_cache[str(product_id)] = scent_description

            logger.info(f"Scent description for product {product_id}: {scent_description}")

        # Save the updated scent cache as a list
        self.save_scent_cache(scent_cache)

        logger.info("All scent descriptions have been updated and saved.")

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

    def query_gpt_for_therapeutic_effect(self, spice_name):
        # spice마다 6개 카테고리(스트레스 감소[1], 행복[2], 리프레시[3], 수면[4], 집중[5], 에너지[6]) 중 어떤 효능이 있는지 또는 관련 없는지[0] GPT에 확인 요청하여 response 저장 (특정 잘 알려진 향료만 추천되는 것을 방지하기 위함)
        prompt = f"""
        Given the perfumery spice "{spice_name}", determine its primary effect among the following categories:
        1. Stress Reduction (스트레스 감소)
        2. Happiness (행복)
        3. Refreshing (리프레시)
        4. Sleep Aid (수면)
        5. Concentration (집중)
        6. Energy Boost (에너지)
        0. Neither
        **If none of these apply, return 0.**
        Respond with only the corresponding number.
        """

        response = self.gpt_client.invoke(prompt).content.strip()

        try:
            return int(response)
        except:
            return 0  # Default to 0 if parsing fails

    def load_json(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def save_json(self, file_path, data):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_spice_therapeutic_effect_cache(self):
        spice_therapeutic_effect_cache_file = self.cache_path_prefix / "spice_therapeutic_effect_cache.json"
        
        spice_data = self.load_cached_spice_data()
        spice_therapeutic_effect_data = self.load_json(spice_therapeutic_effect_cache_file)
        spice_therapeutic_effect_dict = {entry["id"]: entry for entry in spice_therapeutic_effect_data}
        
        updated = False
        for spice in spice_data:
            if spice["id"] not in spice_therapeutic_effect_dict:
                spice_therapeutic_effect_value = self.query_gpt_for_therapeutic_effect(spice["name_en"])
                spice_therapeutic_effect_entry = {"id": spice["id"], "name_en": spice["name_en"], "effect": spice_therapeutic_effect_value}
                spice_therapeutic_effect_data.append(spice_therapeutic_effect_entry)
                spice_therapeutic_effect_dict[spice["name_en"]] = spice_therapeutic_effect_entry
                updated = True
        
        if updated:
            self.save_json(spice_therapeutic_effect_cache_file, spice_therapeutic_effect_data)
            print("spice_therapeutic_effect_cache.json has been updated.")
        else:
            print("All spices already have an entry in spice_therapeutic_effect_cache.json.")

    def load_cached_spice_therapeutic_effect_data(self):
        """Load spice therapeutic effect data from cache."""
        try:
            with open(self.cache_path_prefix / "spice_therapeutic_effect_cache.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("spice_therapeutic_effect_cache.json 파일을 찾을 수 없습니다.")
            return []
        except json.JSONDecodeError:
            logger.error("spice_therapeutic_effect_cache.json 파일을 파싱하는 중 오류가 발생했습니다.")
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
    logger.info("향수 데이터 강제 캐싱 완료!")
