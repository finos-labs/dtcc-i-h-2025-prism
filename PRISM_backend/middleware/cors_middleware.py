from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings

def add_cors_middleware(app: FastAPI) -> None:
    """
    Adds CORS middleware to the FastAPI application
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_origin_regex=r"https://.*\.blockconvey\.com",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "User-Agent", "DNT", "Cache-Control", 
                      "X-Mx-ReqToken", "Keep-Alive", "X-Requested-With", "If-Modified-Since", "X-CSRF-Token", 
                      "Access-Control-Allow-Origin"],
        expose_headers=["Content-Length", "Content-Range"],
        max_age=600,
    ) 