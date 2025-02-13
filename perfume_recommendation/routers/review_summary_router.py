# FastAPI 라우터
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.db_service import get_db
from services.review_summary_service import ReviewService

router = APIRouter()

@router.get("/product/{product_id}/summary")
async def get_review_summary(
    product_id: int, 
    db: Session = Depends(get_db)
):
    """
    제품 리뷰 요약을 가져오는 엔드포인트
    
    Args:
        product_id (int): 제품 ID
        db (Session): 데이터베이스 세션
        
    Returns:
        dict: 요약 정보를 포함한 응답
    """
    review_service = ReviewService()
    summary = await review_service.get_review_summary(str(product_id), db)
    return {"summary": summary}