from typing import List, Literal
from uuid import UUID

from pydantic import BaseModel


class AmazonProduct(BaseModel):
    id: str
    name: str
    price: str
    description: str
    image_url: str
    rating: float
    max_rating: float
    review_count: str
    url: str
    availability: str


class AgentOutput(BaseModel):
    agent_response: str
    amazon_products: List[AmazonProduct]


ProductCategory = Literal[
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
    "Unknown",
]


class SearchProductsInput(BaseModel):
    query: str
    product_category: ProductCategory


class SearchProductsOutput(BaseModel):
    product_asin_array: List[str]
