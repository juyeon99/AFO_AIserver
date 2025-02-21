from fastapi import APIRouter, UploadFile, File, HTTPException
from services.image_processing_service import ImageProcessingService

router = APIRouter()
image_processing_service = ImageProcessingService()


@router.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    업로드된 이미지를 처리하여 설명과 감정을 반환합니다.
    """
    try:
        # 업로드된 파일의 데이터 읽기
        image_data = await file.read()

        # 이미지 처리
        result = image_processing_service.process_image(image_data)

        # 반환값 확인
        if "description" not in result:
            raise HTTPException(status_code=500, detail="🚨 'description' 키가 존재하지 않습니다.")

        # 설명
        return {
            "imageProcessResult": result["description"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")