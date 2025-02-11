import os
import json
import numpy as np
import hashlib  # ✅ md5 해싱을 위한 라이브러리 추가

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # perfume_recommendation 경로
CACHE_DIR = os.path.join(
    BASE_DIR, "embedding_cache"
)  # perfume_recommendation/embedding_cache

os.makedirs(CACHE_DIR, exist_ok=True)  # 폴더가 없으면 생성

def get_cache_filename(identifier: str, cache_type: str) -> str:
    """문자열을 md5 해시로 변환하여 JSON 파일명 생성 (type: image 또는 text)"""
    hashed_identifier = hashlib.md5(identifier.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed_identifier}_{cache_type}.json")

# 🟢 이미지 임베딩 캐싱 기능
def save_embedding(image_url: str, embedding: np.ndarray):
    """JSON 파일(.json)로 이미지 임베딩 저장"""
    filename = get_cache_filename(image_url, "image")  # ✅ 'image' 타입 명시
    with open(filename, "w") as f:
        json.dump(embedding.tolist(), f)
    print(f"💾 이미지 임베딩 저장 완료: {filename}")  # ✅ 저장된 파일 확인

def load_embedding(image_url: str):
    """JSON 파일(.json)에서 이미지 임베딩 불러오기"""
    filename = get_cache_filename(image_url, "image")  # ✅ 'image' 타입 명시
    print(f"🔍 캐시 파일 확인: {filename}")  # ✅ 캐시 파일 확인 메시지

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            print(f"✅ 캐시 불러오기 성공: {filename}")  # ✅ 캐시 성공 메시지
            return np.array(data)

    print(f"❌ 캐시 파일 없음: {filename}")  # ✅ 캐시 파일이 없을 경우
    return None  # 저장된 값이 없으면 None 반환

# 🟢 텍스트 임베딩 캐싱 기능
def save_text_embedding(text: str, embedding: np.ndarray):
    """JSON 파일(.json)로 텍스트 임베딩 저장"""
    filename = get_cache_filename(text, "text")  # ✅ 'text' 타입 명시
    with open(filename, "w") as f:
        json.dump(embedding.tolist(), f)
    print(f"💾 텍스트 임베딩 저장 완료: {filename}")  # ✅ 저장된 파일 확인

def load_text_embedding(text: str):
    """JSON 파일(.json)에서 텍스트 임베딩 불러오기"""
    filename = get_cache_filename(text, "text")  # ✅ 'text' 타입 명시
    print(f"🔍 캐시 파일 확인: {filename}")  # ✅ 캐시 파일 확인 메시지

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            print(f"✅ 캐시 불러오기 성공: {filename}")  # ✅ 캐시 성공 메시지
            return np.array(data)

    print(f"❌ 캐시 파일 없음: {filename}")  # ✅ 캐시 파일이 없을 경우
    return None  # 저장된 값이 없으면 None 반환