# uvicorn scentlens_model_service:app --host 127.0.0.1 --port 8001
from typing import List
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel
from PIL import Image
import io, os, hashlib, torch, requests, logging
from rembg import remove
import numpy as np
from product_schemas import ProductImageItem, ProductImageResult, ProductEmbeddingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)

# 캐시 디렉토리 설정
CACHE_DIR = "../cache/product_image_cache"
EMBEDDING_CACHE_DIR = "../cache/jina_embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

app = FastAPI()

# URL 해시 생성 함수
def generate_hash(url):
    return hashlib.md5(url.encode("utf-8")).hexdigest()

# URL에서 이미지를 다운로드하고 캐싱하는 함수
@app.post("/download_images/", response_model=List[ProductImageResult])
async def download_image_with_cache(product_image_data: List[ProductImageItem]):
    results = []
    for item in product_image_data:
        try:
            filename = os.path.basename(item.url.split("?")[0])  # Remove query parameters
            local_path = os.path.join(CACHE_DIR, filename)

            # 이미 캐시된 이미지인지 확인
            if os.path.exists(local_path):
                logger.info(f"Using cached image: {local_path}")
                results.append({"id": item.id, "url": item.url, "product_id": item.product_id, "local_path": local_path, "status": "success"})
            else:
                logger.info(f"Downloading image: {item.url}")
                response = requests.get(item.url, stream=True)
                response.raise_for_status()

                # 이미지 로컬에 저장
                with open(local_path, "wb") as f:
                    f.write(response.content)

                results.append({"id": item.id, "url": item.url, "product_id": item.product_id, "local_path": local_path, "status": "success"})
        except Exception as e:
            logger.error(f"Failed to process image ID {item.id} from URL {item.url}: {e}")
            results.append({"id": item.id, "url": item.url, "product_id": item.product_id, "error": str(e), "status": "failed"})

    return results

# 임베딩 생성 또는 캐싱
@app.post("/get_or_compute_embeddings/", response_model=List[ProductEmbeddingResult])
async def get_or_compute_embeddings(downloaded_product_image_data: List[ProductImageResult]):
    results = []
    for item in downloaded_product_image_data:
        try:
            # URL 기반으로 해시 생성
            url_hash = generate_hash(item.url)
            embedding_path = os.path.join(EMBEDDING_CACHE_DIR, f"{url_hash}.npy")

            # 캐시된 임베딩 파일이 존재하면 불러오기
            if os.path.exists(embedding_path):
                logger.info(f"Using cached embedding for URL: {item.url}")
                embedding = np.load(embedding_path)
                results.append({"id": item.id, "url": item.url, "product_id": item.product_id, "embedding": embedding.tolist(), "status": "success"})
            else:
                # 캐시된 임베딩 파일이 존재하지 않으면 이미지 다운로드하고 임베딩 계산
                response = requests.get(item.url, stream=True)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

                # 임베딩 생성
                logger.info(f"Computing embedding for URL: {item.url}")
                embedding = model.encode_image(image).flatten()

                # Normalize & 임베딩 저장
                embedding = embedding / np.linalg.norm(embedding)
                np.save(embedding_path, embedding)

                results.append({"id": item.id, "url": item.url, "product_id": item.product_id, "embedding": embedding.tolist(), "status": "success"})
        except Exception as e:
            logger.error(f"Failed to compute embedding for URL {item.url}: {e}")
            results.append({"id": item.id, "url": item.url, "product_id": item.product_id, "error": str(e), "status": "failed"})

    return results

# GPU에서 실행되는 임베딩 생성 함수
@app.post("/compute_embedding_of_uploaded_file/")
async def compute_embedding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 배경 제거
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        output_image_bytes = remove(image_bytes.getvalue())

        output_image = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")

        white_background = Image.new("RGBA", output_image.size, "WHITE")
        white_background.paste(output_image, mask=output_image)
        final_image = white_background.convert("RGB")

        # 임베딩 계산
        embedding = model.encode_image(final_image).flatten()
        embedding = torch.tensor(embedding)
        embedding = embedding / embedding.norm()

        return {"embedding": embedding.cpu().detach().numpy().tolist()}
    except Exception as e:
        return {"error": str(e)}
