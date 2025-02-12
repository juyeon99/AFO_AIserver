from pymongo import MongoClient
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MongoService:
    def __init__(self):
        # MongoDB 연결 설정
        MONGO_URI = "mongodb://banghyang:banghyang@192.168.0.182:27017/banghyang?authSource=banghyang"
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client["banghyang"]

            # 컬렉션 설정
            self.image_embeddings = self.db["image_embeddings"]
            self.text_embeddings = self.db["text_embeddings"]

            # 인덱스 생성
            self.image_embeddings.create_index("identifier", unique=True)
            self.text_embeddings.create_index("identifier", unique=True)

            logger.info("✅ MongoDB 연결 성공")
        except Exception as e:
            logger.error(f"🚨 MongoDB 연결 실패: {e}")
            raise

    def save_image_embedding(self, image_url: str, embedding: np.ndarray):
        """이미지 임베딩을 MongoDB에 저장"""
        try:
            document = {
                "identifier": image_url,
                "embedding": embedding.tolist(),
                "type": "image",
            }
            self.image_embeddings.update_one(
                {"identifier": image_url}, {"$set": document}, upsert=True
            )
            logger.info(f"✅ 이미지 임베딩 저장 완료: {image_url}")
            return True
        except Exception as e:
            logger.error(f"🚨 이미지 임베딩 저장 실패: {e}")
            return False

    def load_image_embedding(self, image_url: str):
        """MongoDB에서 이미지 임베딩 불러오기"""
        try:
            result = self.image_embeddings.find_one({"identifier": image_url})
            if result:
                logger.info(f"✅ 이미지 임베딩 로드 완료: {image_url}")
                return np.array(result["embedding"])
            logger.info(f"❌ 이미지 임베딩 없음: {image_url}")
            return None
        except Exception as e:
            logger.error(f"🚨 이미지 임베딩 로드 실패: {e}")
            return None

    def save_text_embedding(self, text: str, embedding: np.ndarray):
        """텍스트 임베딩을 MongoDB에 저장"""
        try:
            document = {
                "identifier": text,
                "embedding": embedding.tolist(),
                "type": "text",
            }
            self.text_embeddings.update_one(
                {"identifier": text}, {"$set": document}, upsert=True
            )
            logger.info(f"✅ 텍스트 임베딩 저장 완료: {text}")
            return True
        except Exception as e:
            logger.error(f"🚨 텍스트 임베딩 저장 실패: {e}")
            return False

    def load_text_embedding(self, text: str):
        """MongoDB에서 텍스트 임베딩 불러오기"""
        try:
            result = self.text_embeddings.find_one({"identifier": text})
            if result:
                logger.info(f"✅ 텍스트 임베딩 로드 완료: {text}")
                return np.array(result["embedding"])
            logger.info(f"❌ 텍스트 임베딩 없음: {text}")
            return None
        except Exception as e:
            logger.error(f"🚨 텍스트 임베딩 로드 실패: {e}")
            return None

    def __del__(self):
        """소멸자: MongoDB 연결 종료"""
        if hasattr(self, "client"):
            self.client.close()
