
import pytest

from fastapi.testclient import TestClient

import app.main as main


# Fake classifier for testing
class FakeModel:

    def __init__(self, prediction, probability):
        self.prediction = prediction
        self.probability = probability

    def predict(self, texts):
        return [self.prediction]

    def predict_proba(self, texts):
        return [[self.probability, 1 - self.probability]]


@pytest.fixture
def client(monkeypatch):

    # Replace real model loading with fake models
    def fake_load_models():
        return {
            "category": FakeModel("Billing", 0.95),
            "priority": FakeModel("High", 0.90)
        }

    monkeypatch.setattr(
        main,
        "load_models",
        fake_load_models
    )

    with TestClient(main.app) as test_client:
        yield test_client


# Test 1: Health endpoint
def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["models_loaded"] is True


# Test 2: Valid ticket prediction
def test_predict_valid_ticket(client):

    response = client.post(
        "/predict",
        json={"text": "I was charged twice"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Billing"
    assert data["priority"] == "High"
    assert data["confidence"] == 0.925


# Test 3: Empty text should return 422
def test_predict_empty_text(client):

    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


# Test 4: Whitespace-only text should return 422
def test_predict_whitespace(client):

    response = client.post(
        "/predict",
        json={"text": "     "}
    )

    assert response.status_code == 422


# Test 5: Missing text should return 422
def test_predict_missing_text(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


# Test 6: Confidence must be between 0 and 1
def test_confidence_range(client):

    response = client.post(
        "/predict",
        json={"text": "Payment failed"}
    )

    assert response.status_code == 200

    confidence = response.json()["confidence"]

    assert 0 <= confidence <= 1
