import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_scan_file_with_sensitive_data():
    test_text = "My name is John and my phone number is 212-555-7890."
    response = client.post(
        "/scan",
        files={"file": ("test.txt", test_text, "text/plain")}
    )
    assert response.status_code == 200
    result = response.json()
    assert "file" in result
    assert "findings" in result
    assert isinstance(result["findings"], list)
    assert any("entity_type" in finding for finding in result["findings"])

def test_scan_file_with_empty_file():
    response = client.post(
        "/scan",
        files={"file": ("empty.txt", "", "text/plain")}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["findings"] == []
