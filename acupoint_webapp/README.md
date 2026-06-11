# 혈자리 통합 웹앱 (1b · 온디바이스 실시간)

증상 텍스트 → ① LLM 증상분석+KB 혈자리(환각 차단) → 손 카메라 위에 ② **브라우저에서 실시간** 혈자리 표시.
JMIR mHealth 류 사용성 평가용 데모 플랫폼.

## 아키텍처 (왜 1b인가)
```
[브라우저 — 온디바이스]                         [백엔드 — FastAPI(가벼움)]
 카메라 ─ MediaPipe HandLandmarker(WASM) ─ 21랜드마크
        └ M2(JS 포팅) ─ 11혈 좌표 ─ <canvas> AR 오버레이
 증상 입력 ───────────────────── POST /recommend ──▶ ① 증상분석(GPT-4o-mini)+KB
        ◀── 혈자리(+has_cv_model) ── 응급분기 ──────┘
 has_cv_model 혈자리를 손 위에 강조
```
- **CV가 브라우저에서** 돌아 서버는 MediaPipe/OpenGL 불필요. 백엔드는 ①만 담당.
- M2(StandardScaler+MLP, 53→128→128→2)는 작아서 onnxruntime 없이 **순수 JS forward**.
  `static/m2_weights.json`은 `export_m2_weights.py`가 joblib에서 추출(파이썬 `core.py`와 좌표 오차 <1e-3 px 검증됨).

## 구성
```
acupoint_webapp/
├── server.py            # FastAPI: /recommend (① 래핑) + 정적 서빙
├── export_m2_weights.py # joblib -> static/m2_weights.json (1회 실행)
├── run_web.sh, requirements.txt
└── static/
    ├── index.html, style.css
    ├── app.js           # MediaPipe + M2(JS) + AR 오버레이 + 증상연동
    └── m2_weights.json  # M2 가중치(505KB, 재생성 가능)
```

## 실행
```bash
pip install -r requirements.txt
# 키: ../symptom_match/.env.local 의 OPENAI_API_KEY 를 자동 로드(없으면 자동 mock)
./run_web.sh                       # http://localhost:8000
```
브라우저에서 `http://localhost:8000` 접속 → 증상 입력 → ‘카메라 시작’.

- **카메라는 보안 컨텍스트에서만** 동작: `localhost`는 OK. 휴대폰 등 **IP 접속 시 HTTPS 필요**
  (예: `ngrok http 8000` 또는 리버스 프록시 TLS).
- 무과금 테스트: 증상 패널의 `무과금(mock)` 체크(키워드 규칙 — 패러프레이즈는 못 잡음, 정상).

## 검증 상태
- 백엔드 `/recommend`: mock·실제 GPT-4o-mini·응급분기 동작 확인.
- M2 JS 포팅: 실제 landmarks.csv 행으로 파이썬 `core.py` 대비 좌표 오차 5.7e-4 px.
- 브라우저 카메라+MediaPipe+캔버스: `localhost` 데스크톱 브라우저에서 손 검출·혈자리 AR 오버레이 동작 확인 완료.
  (모바일 HTTPS 실기기 테스트는 별도 — 아래 '모바일 테스트' 참고.)

## 모바일 테스트 (HTTPS 필요)
카메라는 보안 컨텍스트(`localhost` 또는 HTTPS)에서만 동작. 휴대폰 등 IP 접속은 TLS 필수.
```bash
./run_web.sh                 # 8000 포트로 로컬 기동(터미널 1)
ngrok http 8000              # 공개 HTTPS URL 발급(터미널 2)
```
발급된 `https://....ngrok-free.app` 를 휴대폰 브라우저로 열어 후면카메라로 손을 비춰 확인.

## 모델/버전 메모
- MediaPipe tasks-vision `0.10.14`(CDN), hand_landmarker float16(Google CDN). 버전 변경은 `app.js`의 `MP_VER`.
- 손 기하 인덱스·chirality 정규화는 `CV_DeepLearning/acupoint_api/core.py`와 동일하게 유지할 것.
