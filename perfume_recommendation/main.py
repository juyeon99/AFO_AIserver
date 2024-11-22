from fastapi import FastAPI
from routes.recommendation_routes import router as recommendation_router
from routes.image_routes import router as image_router
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Perfume Recommendation API",
    description="향수 추천 및 이미지 처리를 제공하는 API입니다.",
    version="1.0.0"
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

# 라우터 등록
app.include_router(recommendation_router, prefix="/recommendations", tags=["Recommendations"])
app.include_router(image_router, prefix="/images", tags=["Image Processing"])

# Uvicorn 실행을 위한 엔트리 포인트
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST , post=APP_PORT)
