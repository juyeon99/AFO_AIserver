import mysql.connector
from mysql.connector import Error
from typing import List, Dict
import logging
from services.prompt_loader import PromptLoader
import json , os
import re

logger = logging.getLogger(__name__)

class DBService:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config

    def some_method_that_needs_gpt_client(self):
        # Lazy Import를 통해 순환 참조 문제 해결
        from models.img_llm_client import GPTClient  
        gpt_client = GPTClient(prompt_loader=PromptLoader("template_path"))

    def fetch_spices_by_line(self, line_name: str) -> List[str]:
        """
        특정 line_name에 속한 향료 목록을 가져옵니다.
        """
        line_query = """
        SELECT id
        FROM line
        WHERE name = %s
        """
        spice_query = """
        SELECT s.name_kr
        FROM spice s
        WHERE s.line_id = %s
        """
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Fetch line_id from line table
            cursor.execute(line_query, (line_name,))
            line_result = cursor.fetchone()
            if not line_result:
                logger.error(f"No line found for line_name: {line_name}")
                return []
            line_id = line_result['id']
            
            # Fetch spices using line_id
            cursor.execute(spice_query, (line_id,))
            spices = [row['name_kr'] for row in cursor.fetchall()]
            logger.info(f"Fetched spices for line_name {line_name} (line_id {line_id}): {spices}")
            return spices
        except Error as e:
            logger.error(f"Database error while fetching spices: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def fetch_perfumes_by_spices(self, spices: List[str]) -> List[Dict]:
        """
        주어진 향료를 기준으로 향수를 가져옵니다.
        """
        placeholders = ', '.join(['%s'] * len(spices))
        query = f"""
        SELECT
            p.id AS perfume_id,
            p.name AS perfume_name,
            p.brand AS perfume_brand,
            p.description AS perfume_description,
            MAX(pi.url) AS perfume_url,
            GROUP_CONCAT(DISTINCT s.name_kr SEPARATOR ', ') AS spice_name,
            COUNT(s.id) AS spice_count
        FROM perfume p
        LEFT JOIN perfume_image pi ON p.id = pi.perfume_id
        LEFT JOIN base_note bn ON p.id = bn.perfume_id
        LEFT JOIN middle_note mn ON p.id = mn.perfume_id
        LEFT JOIN top_note tn ON p.id = tn.perfume_id
        LEFT JOIN spice s ON FIND_IN_SET(s.name_kr, CONCAT_WS(',', bn.spices, mn.spices, tn.spices)) > 0
        WHERE s.name_kr IN ({placeholders})
        GROUP BY p.id
        ORDER BY spice_count DESC
        LIMIT 3;
        """
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, spices)
            perfumes = cursor.fetchall()
            logger.info(f"Fetched perfumes for spices {spices}: {perfumes}")
            return perfumes
        except Error as e:
            logger.error(f"Database error while fetching perfumes: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def fetch_perfumes_by_user_input(self, user_input: str, max_results: int = 3):
        """
        사용자 입력을 기반으로 필터링된 향수 목록을 반환하며, 최대 결과 수를 제한합니다.
        """
        # Load the perfume cache
        base_path = os.path.abspath(os.path.dirname(__file__))
        cache_path = os.path.join(base_path, "..", "perfume_cache.json")

        if not os.path.exists(cache_path):
            raise RuntimeError(f"Perfume cache file not found: {cache_path}")

        with open(cache_path, "r", encoding="utf-8") as file:
            perfumes = json.load(file)

        # Preprocess user input
        user_input = user_input.strip().lower()

        # Use regular expressions to filter perfumes based on user input
        filtered_perfumes = [
            perfume for perfume in perfumes
            if re.search(user_input, perfume["brand"].lower()) or re.search(user_input, perfume["name"].lower())
        ]

        # Log filtered perfumes
        logger.info(f"Filtered perfumes (before limiting results): {filtered_perfumes}")

        # Return limited results
        limited_results = filtered_perfumes[:max_results]
        logger.info(f"Filtered perfumes (limited to {max_results}): {limited_results}")

        if not limited_results:
            raise ValueError(f"No perfumes found for the given user input: {user_input}")

        return limited_results
