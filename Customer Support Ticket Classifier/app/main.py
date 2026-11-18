
from contextlib import asynccontextmanager
from pathlib import Path

import joblib

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Load models only once when the application starts
def load_models():
    category_model = joblib.load(
        MODEL_DIR / "category_classifier.joblib"
    )

    priority_model = joblib.load(
        MODEL_DIR / "priority_classifier.joblib"
    )

    return {
        "category": category_model,
        "priority": priority_model
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    yield


app = FastAPI(
    title="Support Ticket Classification API",
    description="Predict ticket category and priority.",
    version="1.0.0",
    lifespan=lifespan
)


# Request validation
class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Customer support ticket description"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Ticket text cannot be empty")

        return value


# Response schema
class TicketResponse(BaseModel):
    category: str
    priority: str
    confidence: float


@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "models_loaded": hasattr(request.app.state, "models")
    }


@app.post("/predict", response_model=TicketResponse)
def predict_ticket(
    ticket: TicketRequest,
    request: Request
):
    models = request.app.state.models

    text = [ticket.text]

    try:
        # Category prediction
        category_model = models["category"]

        category = category_model.predict(text)[0]

        category_probabilities = (
            category_model.predict_proba(text)[0]
        )

        category_confidence = float(
            max(category_probabilities)
        )

        # Priority prediction
        priority_model = models["priority"]

        priority = priority_model.predict(text)[0]

        priority_probabilities = (
            priority_model.predict_proba(text)[0]
        )

        priority_confidence = float(
            max(priority_probabilities)
        )

        # Combined confidence
        confidence = (
            category_confidence + priority_confidence
        ) / 2

        return {
            "category": str(category),
            "priority": str(priority),
            "confidence": round(confidence, 4)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
