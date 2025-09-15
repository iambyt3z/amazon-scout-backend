import logging

from agents import Runner

from app.agents.amazon_products_search_agent import amazon_products_search_agent
from app.models.amazon_products_search_agent import AgentOutput
from app.models.search_api_models import SearchResponse

logger = logging.getLogger(__name__)


async def search_service(user_message: str) -> SearchResponse:
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
                """,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        logger.info(f"User Message: {user_message}")

        result = await Runner.run(amazon_products_search_agent, messages)
        result_message: AgentOutput = result.final_output
        logger.info(f"Agent Result: {result}")

        return SearchResponse(
            response=result_message.agent_response,
            products=result_message.amazon_products,
        )

    except Exception as e:
        raise e
