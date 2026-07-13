from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict, Optional
from pydantic import BaseModel
import uvicorn

import model_utils
from contextlib import asynccontextmanager
import traceback
from facilities import (
    FacilityRecommendRequest,
    FacilitySearchRequest,
    recommend_facilities,
    search_facilities,
    generate_triage,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, IDX_TO_NAME
    MODEL, IDX_TO_NAME = model_utils.init_model()
    if MODEL is None:
        print("Model not loaded at startup (no TF/model found) — /predict will proxy to MODEL_PROXY_URL if available.")
    else:
        print("Model loaded successfully at startup.")
    yield

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CLIENT_URL = os.getenv("CLIENT_URL")
if CLIENT_URL:
    origins.append(f"{CLIENT_URL}")

app = FastAPI(title="Classifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL = None
IDX_TO_NAME: Dict[int, str] = {}


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await file.read()

    if model_utils.tf is not None and MODEL is not None:
        try:
            x = model_utils.preprocess_image_bytes(data)
            res = model_utils.predict_with_model(MODEL, x)
            pred_idx = res["pred_idx"]
            probs = res["probs"]
            confidence = res["confidence"]
            is_conclusive = res["is_conclusive"]
            pred_name = IDX_TO_NAME.get(pred_idx, str(pred_idx))
            name_prob = {IDX_TO_NAME.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

            if not is_conclusive:
                pred_name = "inconclusive"

            heatmap = model_utils.make_gradcam_heatmap(x, MODEL, res["pred_idx"])
            gradcam_b64 = model_utils.generate_gradcam_base64(data, heatmap)
            gradcam_data_uri = f"data:image/jpeg;base64,{gradcam_b64}"

            triage = generate_triage(
                pred_name if pred_name != "inconclusive" else IDX_TO_NAME.get(res["pred_idx"], str(res["pred_idx"])),
                confidence,
                is_conclusive,
            )

            response = {
                "predicted": pred_name,
                "predicted_idx": pred_idx,
                "confidence": confidence,
                "probabilities": name_prob,
                "gradcam_image": gradcam_data_uri,
                "inconclusive": not is_conclusive,
            }
            if triage:
                response["triage"] = triage

            return response
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Local prediction failed: {e}")

    proxy_url = os.environ.get("MODEL_PROXY_URL")
    if not proxy_url:
        raise HTTPException(status_code=503, detail="Local model unavailable and no proxy configured.")
    try:
        resp = model_utils.proxy_predict(data, getattr(file, "filename", "upload"), file.content_type or "application/octet-stream", proxy_url)
        return resp
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded locally and proxy failed: {e}")


@app.post("/facilities/recommend")
def recommend_facilities_endpoint(body: FacilityRecommendRequest):
    return recommend_facilities(body)


@app.post("/facilities/search")
async def search_facilities_endpoint(body: FacilitySearchRequest):
    return await search_facilities(body)


if __name__ == "__main__":
    print("Starting Classifier API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, loop="asyncio", reload=True)
