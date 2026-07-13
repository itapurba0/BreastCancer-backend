from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import uvicorn

import model_utils

MODEL, IDX_TO_NAME, GRAD_MODEL, NESTED_NAME = model_utils.init_model()
print(f"Model loaded: {MODEL is not None}")

app = FastAPI(title="Grad-CAM API")


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if not file.content_type or file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await file.read()
    if model_utils.tf is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        x = model_utils.preprocess_image_bytes(data)
        res, heatmap = model_utils.predict_with_gradcam(MODEL, GRAD_MODEL, x, NESTED_NAME)
        gradcam_b64 = model_utils.generate_gradcam_base64(data, heatmap)
        return {"gradcam_image": f"data:image/jpeg;base64,{gradcam_b64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
