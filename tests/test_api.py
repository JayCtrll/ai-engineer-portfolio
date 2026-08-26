from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_echo_valid():
    payload = {"message": "Hello", "priority": 3}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    assert response.json()["echo"] == "Echo: Hello"

def test_echo_invalid_priority():
    payload = {"message": "Hello", "priority": 10}
    response = client.post("/echo", json=payload)
    assert response.status_code == 422

def test_secure_data_without_token():
    response = client.get("/secure-data")
    assert response.status_code == 401

def test_secure_data_with_token():
    headers = {"Authorization": "Bearer my-secret-token-123"}
    response = client.get("/secure-data", headers=headers)
    assert response.status_code == 200
    assert "secret" in response.json()
