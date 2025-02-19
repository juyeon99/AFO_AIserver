from pydantic import BaseModel
from typing import Optional, List

class ProductImageItem(BaseModel):
    id: int
    url: str
    product_id: int

class ProductImageResult(BaseModel):
    id: int
    url: str
    product_id: int
    status: str
    local_path: Optional[str] = None
    error: Optional[str] = None

class ProductEmbeddingResult(BaseModel):
    id: int
    url: str
    product_id: int
    status: str
    embedding: Optional[List[float]] = None
    error: Optional[str] = None