from fastapi import APIRouter

from app.api.endpoints import analytics, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
