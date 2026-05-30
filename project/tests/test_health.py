import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from fastapi.testclient import TestClient
 
from src.service.api import app
 
client = TestClient(app)
 
def test_health_status_code():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "recsys-api"


def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
