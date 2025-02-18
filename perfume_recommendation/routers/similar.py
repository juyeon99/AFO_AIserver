from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..services.db_service import get_db, Product, ProductImage
from ..services.similar_text import find_similar_texts
from ..services.similar_image import find_similar_images
from typing import Dict, List

router = APIRouter()


@router.get("/{product_id}")
def get_similar_products(
    product_id: int, db: Session = Depends(get_db), top_n: int = 5
):
    """
    특정 향수에 대해 노트 기반 추천과 디자인 기반 추천을 함께 제공
    """
    # 1. 노트 기반 유사 향수 검색
    note_recommendations = find_similar_texts(product_id, db, top_n)
    note_based = []

    if note_recommendations:
        for rec in note_recommendations:
            product = db.query(Product).filter(Product.id == rec["product_id"]).first()
            if product:
                product_image = (
                    db.query(ProductImage)
                    .filter(ProductImage.product_id == product.id)
                    .first()
                )
                note_based.append(
                    {
                        "id": product.id,
                        "name_kr": product.name_kr,
                        "name_en": product.name_en,
                        "brand": product.brand,
                        "main_accord": product.main_accord,
                        "image_url": product_image.url if product_image else None,
                        "similarity_score": float(rec["similarity"]),
                    }
                )

    # 2. 디자인 기반 유사 향수 검색
    design_recommendations = find_similar_images(product_id, db, top_n)
    design_based = []

    if design_recommendations:
        for rec in design_recommendations:
            product = db.query(Product).filter(Product.id == rec["product_id"]).first()
            if product:
                product_image = (
                    db.query(ProductImage)
                    .filter(ProductImage.product_id == product.id)
                    .first()
                )
                design_based.append(
                    {
                        "id": product.id,
                        "name_kr": product.name_kr,
                        "name_en": product.name_en,
                        "brand": product.brand,
                        "main_accord": product.main_accord,
                        "image_url": product_image.url if product_image else None,
                        "similarity_score": float(rec["similarity"]),
                    }
                )

    return {"note_based": note_based, "design_based": design_based}



