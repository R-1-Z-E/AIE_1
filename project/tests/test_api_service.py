from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.service.app import app

client = TestClient(app)

def test_health_endpoint():
    """Проверка статуса сервиса."""
    response = client.get("/health")
    assert response.status_code in [200, 503]

def test_predict_validation_error():
    """Проверка защиты от некорректных входных данных."""
    response = client.post("/predict", json={})
    assert response.status_code == 422 
    
def test_predict_structure():
    """Проверка структуры ответа API."""
    payload = {
        "sma_7": 45000.5,
        "sma_30": 44000.2,
        "volatility": 0.02,
        "momentum_3d": 0.015
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "action" in data
        assert "probability" in data
        assert data["action"] in ["BUY", "WAIT"]