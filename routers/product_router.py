from fastapi import APIRouter, Depends
from services.product_service import ProductService
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class UserRequest(BaseModel):
    user_content: Optional[str] = None
    image_process_result: Optional[str] = None
    language: Optional[str] = None

def get_product_service():
    return ProductService()

@router.post("/recommend")
async def recommend_product(
    request: UserRequest, 
    product_service: ProductService = Depends(get_product_service)
):
    return product_service.run(request.user_content, request.image_process_result, request.language)
