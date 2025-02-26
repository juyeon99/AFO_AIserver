from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.db_service import get_db, Product, ProductImage
from services.similar_text import find_similar_texts
from services.similar_image import find_similar_images
from concurrent.futures import ThreadPoolExecutor
import asyncio

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)  # 멀티스레딩 설정


async def run_in_threadpool(func, *args):
    """동기 함수를 비동기적으로 실행"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

@router.get("/{product_id}")
async def get_similar_products(
    product_id: int, db: Session = Depends(get_db), top_n: int = 5
):
    """텍스트 기반 및 이미지 기반 유사 향수 추천을 비동기적으로 병렬 실행"""

    # ✅ 비동기 작업 실행 (동시에 실행)
    note_task = run_in_threadpool(find_similar_texts, product_id, top_n)
    design_task = run_in_threadpool(find_similar_images, product_id, top_n)

    # ✅ 두 작업을 병렬 실행하여 결과 받기
    note_recommendations, design_recommendations = await asyncio.gather(
        note_task, design_task
    )

    # ✅ 결과 변환 (데이터베이스 접근 최소화)
    def transform_result(recommendations):
        return [
            {
                "id": rec["product_id"],
                "name_kr": product.name_kr,
                "name_en": product.name_en,
                "brand": product.brand,
                "main_accord": product.main_accord,
                "image_url": product_image.url if product_image else None,
                "similarity_score": float(rec["similarity"]),
            }
            for rec in recommendations
            if (
                product := db.query(Product)
                .filter(Product.id == rec["product_id"])
                .first()
            )
            is not None
            and (
                product_image := db.query(ProductImage)
                .filter(ProductImage.product_id == product.id)
                .first()
            )
            is not None
        ]

    return {
        "note_based": transform_result(note_recommendations),
        "design_based": transform_result(design_recommendations),
    }
