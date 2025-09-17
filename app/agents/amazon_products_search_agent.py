import logging
import os
import re

import requests
from agents import Agent, ModelSettings, function_tool
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from app.classes.dense_search_client import DenseSearchClient
from app.classes.hybrid_search_client import HybridSearchClient
from app.models.amazon_products_search_agent import (
    AgentOutput,
    AmazonProduct,
    SearchProductsInput,
    SearchProductsOutput,
)

load_dotenv()

logger = logging.getLogger(__name__)
hybrid_search_client = HybridSearchClient()

dense_search_client = DenseSearchClient(
    collection_name=os.getenv("QDRANT_COLLECTION", "hybrid-search2"),
    ensure_schema=False,  # set True if you want the client to (re)create the collection schema
    deterministic=False,
)


@function_tool
def fetch_amazon_product_details(asin: str, domain: str = "amazon.com") -> AmazonProduct:
    """
    Fetch amazon product's details like name, price, ratings, etc.
    This tool returns the amazon product details.

    Args:
        asin: Amazon product id (asin)
        domain: Domain of amazon's website, Default value is "amazon.com"
    """
    url = f"https://www.{domain}/dp/{asin}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        product_details = {
            "asin": asin,
            "url": url,
            "name": None,
            "image": None,
            "price": None,
            "rating": None,
            "review_count": None,
            "availability": None,
            "brand": None,
            "description": None,
        }

        name_selectors = [
            "#productTitle",
            ".product-title",
            "h1.a-size-large",
            "a-size-large product-title-word-break",
        ]

        for selector in name_selectors:
            name_element = soup.select_one(selector)

            if name_element:
                product_details["name"] = name_element.get_text().strip()
                break

        image_selectors = [
            "#landingImage",
            ".a-dynamic-image",
            "#imgTagWrapperId img",
            ".a-button-thumbnail img",
        ]

        for selector in image_selectors:
            image_element = soup.select_one(selector)
            if image_element:
                image_url = image_element.get("src") or image_element.get("data-src")
                if image_url:
                    product_details["image"] = image_url
                    break

        price_selectors = [
            ".a-price-whole",
            ".a-offscreen",
            "#priceblock_dealprice",
            "#priceblock_ourprice",
            ".a-price .a-offscreen",
        ]

        for selector in price_selectors:
            price_element = soup.select_one(selector)
            if price_element:
                price_text = price_element.get_text().strip()
                if "$" in price_text or "₹" in price_text or "€" in price_text:
                    product_details["price"] = price_text
                    break

        rating_selectors = [
            ".a-icon-alt",
            '[data-hook="average-star-rating"] .a-icon-alt',
            ".reviewCountTextLinkedHistogram .a-icon-alt",
        ]

        for selector in rating_selectors:
            rating_element = soup.select_one(selector)
            if rating_element:
                rating_text = rating_element.get_text()
                rating_match = re.search(r"(\d+\.?\d*)\s*out of", rating_text)
                if rating_match:
                    product_details["rating"] = rating_match.group(1)
                    break

        review_selectors = [
            '[data-hook="total-review-count"]',
            ".a-link-normal .a-size-base",
            "#acrCustomerReviewText",
        ]

        for selector in review_selectors:
            review_element = soup.select_one(selector)
            if review_element:
                review_text = review_element.get_text()
                review_match = re.search(r"([\d,]+)", review_text)
                if review_match:
                    product_details["review_count"] = review_match.group(1)
                    break

        availability_selectors = [
            "#availability span",
            ".a-size-medium.a-color-success",
            ".a-size-medium.a-color-price",
        ]

        for selector in availability_selectors:
            availability_element = soup.select_one(selector)
            if availability_element:
                availability_text = availability_element.get_text().strip()
                if availability_text and len(availability_text) < 100:
                    product_details["availability"] = availability_text
                    break

        feature_bullets = soup.select("#feature-bullets ul li")
        if feature_bullets:
            features = []
            for bullet in feature_bullets[:3]:
                text = bullet.get_text().strip()
                if text and not text.startswith("Make sure"):
                    features.append(text)
            if features:
                product_details["description"] = "; ".join(features)

        return AmazonProduct(
            availability=product_details.get("availability", ""),
            description=product_details.get("description", ""),
            id=product_details.get("asin", ""),
            image_url=product_details.get("image", ""),
            max_rating=5.0,
            name=product_details.get("name", ""),
            price=product_details.get("price", ""),
            rating=float(product_details.get("rating", "")),
            review_count=product_details.get("review_count", ""),
            url=url,
        )

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": f"Request failed: {e}", "asin": asin, "url": url}

    except Exception as e:
        print(f"Parsing failed: {e}")
        return {"error": f"Parsing failed: {e}", "asin": asin, "url": url}


@function_tool
async def search_products(search_products_input: SearchProductsInput) -> SearchProductsOutput:
    """
    Search for amazon products matching the user query.
    This tool returns the amazon product ids (asin).

    Args:
        search_products_input:
            Input containing search query for the product the user wants to search and
            category of the product.
    """
    try:
        query = search_products_input.query
        category = search_products_input.product_category

        # filtered_results = client.hybrid_search(query, topk=25, filters={"category": category})
        # filtered_results = hybrid_search_client.hybrid_search(query, topk=25)
        filtered_results = dense_search_client.search(query, topk=25, filters={})

        product_asin_array = []

        for i, point in enumerate(filtered_results.points, 1):
            product_asin = point.payload.get("asin", "N/A")
            product_asin_array.append(product_asin)

        return SearchProductsOutput(product_asin_array=product_asin_array)

    except Exception as e:
        logger.exception(e)
        raise Exception(f"Search amazon products failed with error: {e}")


instructions = """
You are a SEARCH AMAZON PRODUCTS agent designed to help users find relevant Amazon products based on their queries. 
Your primary role is to understand user intent, search for appropriate products, and provide detailed, helpful product information.

Core Responsibilities:

    1. Query Understanding: Analyze user queries to extract:
        * Product type/category
        * Key features or specifications
        * Price preferences
        * Brand preferences
        * Use case or purpose

    2. Product Search: Use the search_products tool to find relevant products matching user criteria

    3. Product Details: Fetch comprehensive product information including names, ratings, prices, links, and key features

    4. Result Presentation: Present findings in a clear, organized manner that helps users make informed decisions

User Flow:

    1. User will send one message which will be the query to find the product

    2. You will then figure out the product category from the user query

    3. You will then use the query and product category to fetch the relevant products
    using the 'search_products' tool. This will give you the asin ids of all the relevant
    products

    4. You will then use the asin ids which you got previously to fetch the product details
    using the 'fetch_amazon_product_details' tool. This will give you the product details
    such as name, id, url, etc.

    5. Finally return the product details to the user.

EDGE CASES RULES:

    1. If search returns no results, suggest alternative search terms or broader categories.

    2. If product details are incomplete, present available information clearly.   

PRODUCT CATEGORY RULES:

    1. when you are figuring out the product category from the query, make sure the value is always from these values:
        "All_Beauty",
        "Amazon_Fashion",
        "Appliances",
        "Arts_Crafts_and_Sewing",
        "Automotive",
        "Baby_Products",
        "Beauty_and_Personal_Care",
        "Books",
        "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",
        "Electronics",
        "Gift_Cards",
        "Grocery_and_Gourmet_Food",
        "Handmade_Products",
        "Health_and_Household",
        "Health_and_Personal_Care",
        "Home_and_Kitchen",
        "Industrial_and_Scientific",
        "Kindle_Store",
        "Magazine_Subscriptions",
        "Movies_and_TV",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Software",
        "Sports_and_Outdoors",
        "Subscription_Boxes",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games",
        "Unknown"

OUTPUT FORMAT:

{
    "agent_response": STR // Your general response to the user's query
    "amazon_products": [
        {
            "id": str // Product id (asin)
            "name": str // Name of the product
            "price": str // Price of the product
            "description": str // Description of the product
            "image_url": str // URL of the product image
            "rating": int // Overall user rating of the product
            "max_rating": int // Maximum rating of the product (always 5)
            "review_count": str // Number of reviews posted for the product
            "url": str // URL Link of the product
            "availability": str // String explaining the availability of the product
        }
    ]
}
"""

settings = ModelSettings(
    max_tokens=10000,
    temperature=0.3,
    parallel_tool_calls=True,
    tool_choice="required",
)

amazon_products_search_agent = Agent(
    name="Amazon Product Search Agent",
    instructions=instructions,
    model_settings=settings,
    model="gpt-4.1",
    output_type=AgentOutput,
    tools=[
        search_products,
        fetch_amazon_product_details,
    ],
)
