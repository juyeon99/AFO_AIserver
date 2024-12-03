from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.llm_service import LLMService
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from models.img_llm_client import GPTClient
from utils.line_mapping import LineMapping
import os , logging

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Dependency initialization
template_path = os.path.join(os.path.dirname(__file__), "..", "models", "prompt_template.json")
line_file_path = os.path.join(os.path.dirname(__file__), "..", "models", "line.json")
line_mapping = LineMapping(line_file_path)

# Load environment variables for API key and database config
api_key = os.getenv("OPENAI_API_KEY")
db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# Validate configuration
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
if not all(db_config.values()):
    raise RuntimeError("Incomplete database configuration. Please check environment variables.")

# Create dependency instances
db_service = DBService(db_config=db_config)
prompt_loader = PromptLoader(template_path)
gpt_client = GPTClient(prompt_loader=prompt_loader)

# Create LLMService instance
llm_service = LLMService(
    gpt_client=gpt_client,
    db_service=db_service,
    prompt_loader=prompt_loader,
    line_mapping=line_mapping
)

# Define Pydantic model
class UserInput(BaseModel):
    user_input: str

def get_llm_service() -> LLMService:
    return llm_service

@router.post("/process-input")
async def process_user_input(input_data: UserInput, llm_service: LLMService = Depends(get_llm_service)):
    """
    Process user input and return either conversation or perfume recommendation results.
    """
    try:
        user_input = input_data.user_input
        mode, line_id = llm_service.process_input(user_input)

        logger.info(f"Processing input with mode: {mode}, line_id: {line_id}")

        if mode == "chat":
            response = llm_service.generate_chat_response(user_input)
            return {"mode": "chat", "response": response}

        elif mode == "recommendation":
            if line_id is None:
                raise HTTPException(status_code=400, detail="Line ID not found for recommendation.")
            response = llm_service.generate_recommendation_response(user_input, line_id)

            # Translate image prompt to English if needed
            image_prompt = response["image_prompt"]
            translated_image_prompt = gpt_client.generate_response(f"Translate the following text to English:\n{image_prompt}")

            return {
                "mode": "recommendation",
                "recommended_perfumes": response["recommendation"],
                "common_feeling": response["common_feeling"],
                "image_prompt": translated_image_prompt
            }

        else:
            raise HTTPException(status_code=400, detail="Unknown mode")

    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Input Error: {str(e)}")

    except RuntimeError as e:
        logger.error(f"RuntimeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service Error: {str(e)}")

    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
