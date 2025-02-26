from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import requests, faiss, json, torch, io, os, logging
import numpy as np
from services.db_service import DBService

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 로그
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB, FAISS 인덱스 초기화
db_images = []
db_embeddings = []
index = None
product_data = []

router = APIRouter()

# 서버 시작 전 미리 실행할 코드; 서버를 initialize하여 데이터 로드, 이미지 다운로드, 임베딩 계산, FAISS 인덱스 생성을 미리 수행
def scentlens_init():
    global db_images, db_embeddings, index, product_data

    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
    }
    db_service = DBService(db_config)

    # JSON 데이터 로드
    product_image_data = db_service.load_cached_product_image_data()
    perfume_data = db_service.load_cached_perfume_data()
    diffuser_data = db_service.load_cached_diffuser_data()
    product_data = perfume_data + diffuser_data

    if not product_image_data or not product_data:
        logger.error("Initialization failed due to missing or invalid data.")
        return

    # 이미지 다운로드
    downloaded_images = download_images(product_image_data)
    if not downloaded_images:
        logger.error("Initialization failed due to image downloading errors.")
        return

    # 임베딩 계산
    embeddings_data = compute_embeddings(downloaded_images)
    if not embeddings_data:
        logger.error("Initialization failed due to embedding computation errors.")
        return

    # DB에 이미지 및 임베딩 저장
    populate_db(embeddings_data)

    # FAISS 인덱스 생성
    create_faiss_index()

# 이미지 다운로드를 위한 배치 요청을 전송하고 응답을 반환
def download_images(product_image_data):
    try:
        response = requests.post("http://localhost:8001/download_images/", json=product_image_data)
        if response.status_code == 200:
            logger.info("Successfully downloaded images.")
            return response.json()
        else:
            logger.error(f"Failed to download images. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error during image downloading: {e}")
        return None

# 다운로드된 이미지에 대해 임베딩 계산을 위한 배치 요청을 전송
def compute_embeddings(downloaded_images):
    try:
        response = requests.post("http://localhost:8001/get_or_compute_embeddings/", json=downloaded_images)
        if response.status_code == 200:
            logger.info("Successfully computed embeddings.")
            return response.json()
        else:
            logger.error(f"Failed to compute embeddings. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error during embedding computation: {e}")
        return None

# DB에 이미지 및 임베딩 데이터 저장
def populate_db(embeddings_data):
    global db_images, db_embeddings

    for item in embeddings_data:
        if item["status"] == "success":
            db_images.append({"id": item["id"], "url": item["url"], "product_id": item["product_id"]})
            embedding = torch.tensor(item["embedding"])
            db_embeddings.append(embedding)
        else:
            logger.error(f"Failed to process embedding for image ID {item['id']} from URL {item['url']}: {item['error']}")

# 계산된 임베딩을 사용하여 FAISS 인덱스를 생성
def create_faiss_index():
    global db_embeddings, index

    if db_embeddings:
        db_embeddings = torch.stack(db_embeddings)
        db_embeddings = db_embeddings / db_embeddings.norm(dim=1, keepdim=True)

        dimension = db_embeddings.shape[1]
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dimension)
        index.add(db_embeddings.numpy())
        logger.info("FAISS index created successfully.")
    else:
        logger.info("No embeddings available. Initializing an empty FAISS index.")
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), 1)

# 임베딩값으로 향수 매칭
def get_matching_products(embedding, db_images, db_embeddings, product_data, threshold=0.3, k=10, max_results=10):
    # FAISS 인덱스에서 검색
    results = []
    product_ids_set = set()  # 이미 처리된 제품 ID를 추적
    batch = k * 2

    while len(product_ids_set) < max_results:
        D, I = index.search(np.array(embedding).reshape(1, -1), k=batch)
        
        for idx, i in enumerate(I[0]):
            similarity = float(D[0][idx])
            if similarity > threshold:
                product_id = db_images[i]["product_id"]

                # 이미 해당 제품이 결과에 포함되었는지 확인
                if product_id not in product_ids_set:
                    product_ids_set.add(product_id)

                    results.append({
                        "index": int(i),
                        "id": db_images[i]["id"],
                        "product_id": product_id,
                        "url": db_images[i]["url"],
                        "similarity": similarity,
                    })

                    # 결과가 충분히 모였으면 종료
                    if len(product_ids_set) >= max_results:
                        break

        batch += k

        # 결과가 부족하면 더 많은 검색을 진행
        if max_results > len(product_ids_set):
            continue
        else:
            break

    # 해당 제품의 상세 정보를 가져옴
    matching_products = [
        {
            "id": item["id"],
            "name": item["name_kr"],
            "brand": item["brand"],
            "content": item["content"],
            "similarity": next(
                (result["similarity"] for result in results if result["product_id"] == item["id"]), None
            ),
            "url": next(
                (result["url"] for result in results if result["product_id"] == item["id"]), None
            ),
        }
        for item in product_data if item["id"] in product_ids_set
    ]

    # 유사도 기준으로 내림차순 정렬
    return sorted(matching_products, key=lambda x: x["similarity"], reverse=True)[:max_results]

@router.post("/get_image_search_result")
async def search_image(file: UploadFile = File(...)):
    try:
        # GPU 임베딩 서비스 호출
        image_bytes = await file.read()

        compute_url = "http://localhost:8001/compute_embedding_of_uploaded_file/"
        response = requests.post(
            compute_url, files={"file": ("uploaded_image.png", image_bytes)}
        )

        if response.status_code == 200:
            embedding = response.json().get("embedding")
            
            if embedding is not None:
                matching_products = get_matching_products(embedding, db_images, db_embeddings, product_data)

                return {"products": sorted(matching_products, key=lambda x: x["similarity"], reverse=True)}
            else:
                return {"error": "No embedding found in the response"}
        else:
            return {"error": f"Failed to get embedding. Status code: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error retrieving product details: {e}")
        return {"error": str(e)}
