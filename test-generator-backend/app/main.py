"""
A4AI Test Generator Backend — FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints.test_generator import router as test_router
from app.api.v1.endpoints.contest import router as contest_router
from app.api.v1.endpoints.payment import router as payment_router
from app.api.v1.endpoints.community_quiz import router as community_quiz_router
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="A4AI Test Generator API",
    description="AI-powered test generation for K-12 teachers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(test_router, prefix="/api/v1")
app.include_router(contest_router, prefix="/api/v1")
app.include_router(payment_router, prefix="/api/v1")
app.include_router(community_quiz_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "a4ai-test-generator"}


@app.get("/")
def root():
    return {"message": "A4AI Test Generator API v1.0"}