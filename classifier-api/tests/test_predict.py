import io
import os
import sys

import pytest # type: ignore
from fastapi.testclient import TestClient # type: ignore


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app as backend_api


@pytest.fixture(autouse=True)
def patch_proxy_predict(monkeypatch):
   
    def fake_proxy(file_bytes, filename, content_type, proxy_url):
        return {"predicted": "benign", "predicted_idx": 1, "probabilities": {"benign": 0.9}}

    monkeypatch.setattr(backend_api.model_utils, "proxy_predict", fake_proxy)
   
    backend_api.MODEL = None
    yield


def test_predict_endpoint_returns_json():
    client = TestClient(backend_api.app)

   
    fake_file = io.BytesIO(b"fake-image-bytes")
    files = {"file": ("test.png", fake_file, "image/png")}

    resp = client.post("/predict", files=files)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "predicted" in data
    assert data["predicted"] in ("benign", "malignant", "normal",) or isinstance(data["predicted"], str)
