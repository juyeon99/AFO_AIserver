from fastapi import APIRouter, Depends
from services.perfume_service import PerfumeService
from pydantic import BaseModel

router = APIRouter()

class UserRequest(BaseModel):
    user_input: str

def get_perfume_service():
    return PerfumeService()

@router.post("/recommend")
async def recommend_perfume(
    request: UserRequest, 
    perfume_service: PerfumeService = Depends(get_perfume_service)
):
    return perfume_service.run(request.user_input)
