"""
혈자리 추론 FastAPI 서버.

엔드포인트:
  GET  /health            상태 확인
  GET  /acupoints         지원 혈자리 메타 목록
  POST /predict           손 사진 업로드 -> 혈자리 좌표 JSON
  POST /predict/overlay   손 사진 업로드 -> 혈자리 표시된 PNG 반환

실행:  ./run_api.sh   (또는 LD_LIBRARY_PATH 설정 후 uvicorn server:app)
"""
import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse

from core import AcupointPredictor, ACUP_META

app = FastAPI(title="Acupoint Inference API", version="0.1.0")
predictor: AcupointPredictor | None = None


@app.on_event("startup")
def _load():
    global predictor
    predictor = AcupointPredictor()


def _read_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다.")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None,
            "n_acupoints": len(predictor.acups) if predictor else 0}


@app.get("/acupoints")
def acupoints():
    return [{"name": k, "name_kr": v[0], "code": v[1]} for k, v in ACUP_META.items()]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = _read_image(await file.read())
    return JSONResponse(predictor.predict(img))


@app.post("/predict/overlay")
async def predict_overlay(file: UploadFile = File(...)):
    img = _read_image(await file.read())
    result = predictor.predict(img)
    vis = predictor.overlay(img, result)
    ok, buf = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="오버레이 인코딩 실패")
    return Response(content=buf.tobytes(), media_type="image/png")
