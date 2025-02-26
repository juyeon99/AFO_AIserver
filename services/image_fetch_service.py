import os
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.responses import Response
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class ImageFetchService:
    def __init__(self):
        pass

    def get_image(self, image_path: str):
        # Ensure the image exists
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        # Read the image as raw bytes
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # image = Image.open(io.BytesIO(image_bytes))
        # image.show()

        # Return the raw bytes as a response with correct media type
        return Response(content=image_bytes, media_type="image/jpeg")