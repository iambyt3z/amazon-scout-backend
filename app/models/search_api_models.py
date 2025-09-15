from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel

from app.models.amazon_products_search_agent import AmazonProduct


class SearchRequest(BaseModel):
    message: str


class SearchResponse(BaseModel):
    response: str
    products: List[AmazonProduct]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
