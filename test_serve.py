# test_serve.py

from fastapi.testclient import TestClient
from serve import app

# Create a TestClient to test the FastAPI app
client = TestClient(app)

def test_predict_valid_claim():
    """Test that a valid claim returns a successful response with a label."""
    response = client.post(
        "/claim/v1/predict",
        json={"claim": "Some example health-related claim"}
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert isinstance(response.json()["label"], int)

def test_predict_empty_claim():
    """Test that an empty claim returns a valid response with a label."""
    response = client.post(
        "/claim/v1/predict",
        json={"claim": ""}
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert isinstance(response.json()["label"], int)

def test_predict_missing_claim_field():
    """Test that a missing claim field returns a 422 error."""
    response = client.post(
        "/claim/v1/predict",
        json={}
    )
    assert response.status_code == 422

def test_predict_non_string_claim():
    """Test that a non-string claim returns a valid response with a label."""
    response = client.post(
        "/claim/v1/predict",
        json={"claim": 1234}
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert isinstance(response.json()["label"], int)
