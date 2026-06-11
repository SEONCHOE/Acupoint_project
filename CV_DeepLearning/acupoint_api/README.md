# Acupoint Inference API

손 사진 한 장 → MediaPipe 21 랜드마크 → M2 모델 → **11개 혈자리 좌표**.
앱 백엔드의 추론 코어. `core.py`(재사용 모듈) + `server.py`(FastAPI).

## 구조
```
acupoint_api/
├── core.py            # AcupointPredictor — 어디서든 재사용하는 추론 진입점
├── server.py          # FastAPI: /health /acupoints /predict /predict/overlay
├── run_api.sh         # LD_LIBRARY_PATH(OpenGL) 세팅 + uvicorn 실행
├── requirements.txt
└── models/
    ├── hand_landmarker.task   # MediaPipe 손 랜드마크 모델
    └── m2_model.joblib        # 학습된 랜드마크→혈자리 MLP (+혈자리 목록)
```

## 모듈로 직접 사용
```python
from core import AcupointPredictor
import cv2

predictor = AcupointPredictor()
result = predictor.predict(cv2.imread("hand.jpg"))
# result = {detected, image_size:[w,h], handedness, landmarks:[[x,y]*21],
#           acupoints:[{name, name_kr, code, x, y} * 11]}
overlay = predictor.overlay(cv2.imread("hand.jpg"), result)  # 시각화 BGR ndarray
```

## 서버 실행
```bash
./run_api.sh                       # http://0.0.0.0:8000
# 또는: uvicorn server:app --port 8000   (LD_LIBRARY_PATH 선설정 필요)
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 상태·모델 로드 여부 |
| `GET /acupoints` | 지원 11혈 메타(한글명·경혈코드) |
| `POST /predict` | 이미지 업로드(multipart `file`) → 혈자리 좌표 JSON |
| `POST /predict/overlay` | 이미지 업로드 → 혈자리 표시 PNG |

```bash
curl -X POST http://localhost:8000/predict -F "file=@hand.jpg"
curl -X POST http://localhost:8000/predict/overlay -F "file=@hand.jpg" -o overlay.png
```

## 배포 주의 (OpenGL 의존성)
MediaPipe는 `libGLESv2.so.2`, `libEGL.so.1`를 요구한다. 정식 환경에서는:
```bash
sudo apt install -y libgles2 libegl1 libglib2.0-0
```
현재 개발 WSL에는 root 없이 `/tmp/gllibs`에 추출해 `run_api.sh`가 그 경로를 참조함
(재부팅 시 사라지므로 영구 설치 권장). 자세한 내용은 메모리 `mediapipe-env-setup` 참고.

## 성능 (참고)
M2 모델: test PCK@0.10 = 96.1%, 픽셀 중앙오차 8.2px(700px 이미지). 현행 ResNet 대비 ekmoon 2.7배 우수.
검출률은 손 전체가 보이는 이미지 기준 ~85~90%(크롭/블러 손은 미검출 → 앱에서 재촬영 안내 권장).
```
