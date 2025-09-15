import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import search
from app.core.logging import setup_logging

load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="Amazon Product Search Backend", version="1.0.0")

    setup_logging()

    # Configure CORS (optional, only if needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*", "http://localhost:3000/"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # fmt: off
    # Include routers
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(search.router, prefix="/v1/search", tags=["Amazon Product Search"])
    # fmt: on

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"

    print(f"Starting server on {host}:{port}")

    uvicorn.run("main:app", host=host, port=port, reload=True)
