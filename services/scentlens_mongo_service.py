import atexit
from pymongo import MongoClient
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongouri = os.getenv("MONGO_URI")

class ScentlensMongoService:
    def __init__(self):
        # MongoDB 연결 설정
        MONGO_URI = mongouri

        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client["banghyang"]

            # 컬렉션 설정
            self.scentlens_image_embeddings = self.db["scentlens_image_embeddings"]

            # 인덱스 생성
            self.scentlens_image_embeddings.create_index("url", unique=True)

            logger.info("✅ MongoDB 연결 성공 (ScentLens)")

            atexit.register(self.close_connection)
        except Exception as e:
            logger.error(f"🚨 MongoDB 연결 실패: {e}")
            raise

    def save_jina_image_embedding(self, id: int, image_url: str, product_id: int, embedding: np.ndarray):
        """센트렌즈에 필요한 이미지 임베딩을 MongoDB에 저장"""
        try:
            document = {
                "id": id,
                "url": image_url,
                "product_id": product_id,
                "embedding": embedding.tolist()
            }

            self.scentlens_image_embeddings.update_one(
                {"url": image_url}, {"$set": document}, upsert=True
            )

            logger.info(f"✅ 이미지 임베딩 저장 완료: {id}")
            return True
        except Exception as e:
            logger.error(f"🚨 이미지 임베딩 저장 실패: {e}")
            return False
    
    def load_jina_image_embedding(self, url: str):
        """MongoDB에서 센트렌즈에 필요한 이미지 임베딩 불러오기"""
        try:
            result = self.scentlens_image_embeddings.find_one({"url": url})

            if result:
                logger.info(f"✅ Jina 이미지 임베딩 로드 완료: {url}")
                return np.array(result["embedding"])
            
            logger.info(f"⚠ Jina 이미지 임베딩 없음: {url}")

            return None
        except Exception as e:
            logger.error(f"🚨 Jina 이미지 임베딩 로드 실패: {e}")
            return None
    
    def delete_jina_image_embedding(self, url: str):
        """MongoDB에서 특정 URL의 이미지 임베딩 삭제"""
        try:
            result = self.scentlens_image_embeddings.delete_one({"url": url})
            if result.deleted_count > 0:
                logger.info(f"✅ 이미지 임베딩 삭제 완료: {url}")
                return True
            logger.info(f"❌ 삭제할 이미지 임베딩 없음: {url}")
            return False
        except Exception as e:
            logger.error(f"🚨 이미지 임베딩 삭제 실패: {e}")
            return False

    def clear_jina_image_embeddings(self):
        """MongoDB에서 모든 이미지 임베딩 삭제 (컬렉션 비우기)"""
        try:
            result = self.scentlens_image_embeddings.delete_many({})
            logger.info(f"✅ 모든 이미지 임베딩 삭제 완료: {result.deleted_count}개 삭제됨")
            return True
        except Exception as e:
            logger.error(f"🚨 이미지 임베딩 컬렉션 비우기 실패: {e}")
            return False
    
    def close_connection(self):
        """MongoDB 연결 종료"""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            logger.info("✅ MongoDB 연결 종료 (ScentLens)")
