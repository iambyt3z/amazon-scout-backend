import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.models.search_api_models import SearchRequest, SearchResponse
from app.services.search_services import search_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=SearchResponse)
async def chat(request: SearchRequest):
    try:
        user_message = request.message

        response = await search_service(user_message)

        return response

    except Exception as e:
        logger.exception(f"Unexpected error while creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error"
        )
