import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from perfume_recommendation.routers import llm_router, image_processing_router, image_generation_router, image_generation_description_router, diffuser_router, similar, review_summary_router, bookmark_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles


# 환경 변수 로드
load_dotenv()

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Perfume Recommendation API",
    description="향수 추천 및 이미지 처리를 제공하는 API입니다.",
    version="1.0.1"
)

APP_HOST = os.getenv("APP_HOST")
APP_PORT = int(os.getenv("APP_PORT"))

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 접근 허용 (프로덕션 환경에서는 제한 필요)
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리 생성
os.makedirs("generated_images", exist_ok=True)

# static 파일 서빙 설정 - 경로 수정
app.mount("/static", StaticFiles(directory="generated_images"), name="static")

app.include_router(llm_router.router, prefix="/llm", tags=["LLM"])
app.include_router(image_processing_router.router, prefix="/image-processing", tags=["Image Processing"])
app.include_router(image_generation_router.router, prefix="/image-generation", tags=["Image Generation"])
app.include_router(image_generation_description_router.router, prefix="/llm" , tags=["LLM-Image-Description"])
app.include_router(diffuser_router.router, prefix="/diffuser", tags=["Diffuser"])
app.include_router(similar.router, prefix="/similar", tags=["Similar"])
app.include_router(review_summary_router.router, prefix="/review", tags=["Review"])
app.include_router(bookmark_router.router, prefix="/bookmark", tags=["Bookmark"])

# Uvicorn 실행을 위한 엔트리 포인트
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)