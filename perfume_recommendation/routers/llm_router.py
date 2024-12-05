from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.llm_service import LLMService
from services.db_service import DBService
from services.prompt_loader import PromptLoader
from models.img_llm_client import GPTClient
from utils.line_mapping import LineMapping
import os
import logging

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Dependency initialization function
def get_llm_service() -> LLMService:
    try:
        template_path = os.path.join(os.path.dirname(__file__), "..", "models", "prompt_template.json")
        line_file_path = os.path.join(os.path.dirname(__file__), "..", "models", "line.json")

        # Load environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
        if not all(db_config.values()):
            raise RuntimeError("Incomplete database configuration. Please check environment variables.")

        # Create instances
        line_mapping = LineMapping(line_file_path)
        db_service = DBService(db_config=db_config)
        prompt_loader = PromptLoader(template_path)
        gpt_client = GPTClient(prompt_loader=prompt_loader)

        return LLMService(
            gpt_client=gpt_client,
            db_service=db_service,
            prompt_loader=prompt_loader,
            line_mapping=line_mapping
        )
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize services.")

# Pydantic model for input validation
class UserInput(BaseModel):
    user_input: str

@router.post("/process-input")
async def process_user_input(input_data: UserInput, llm_service: LLMService = Depends(get_llm_service)):
    """
    Process user input and return either conversation or perfume recommendation results.
    """
    try:
        user_input = input_data.user_input
        mode, line_id = llm_service.process_input(user_input)

        logger.info(f"Processing user input: mode={mode}, input={user_input}")

        if mode == "chat":
            response = llm_service.generate_chat_response(user_input)
            return {"mode": "chat", "response": response}

        elif mode == "recommendation":
            if not line_id:
                logger.error("Line ID not found for recommendation.")
                raise HTTPException(status_code=400, detail="Line ID not found for recommendation.")

            # Generate recommendation response
            response = llm_service.generate_recommendation_response(user_input, line_id)

            # Extract and format response fields
            recommendations = response.get("recommendations", [])
            content = response.get("content", "No common feeling generated.")
            line_id = response.get("line_id", "No line_id generated")

            return {
                "mode": "recommendation",
                "recommendations": recommendations,
                "content": content,
                "line_id": line_id
            }

        else:
            raise HTTPException(status_code=400, detail="Unknown mode")

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e

    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Input Error: {str(e)}")

    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")