# 배포 — Render (모바일 HTTPS)

`acupoint_webapp`을 Render 무료 웹 서비스로 배포한다. HTTPS 자동 → 휴대폰에서 카메라·음성(STT) 동작.
설정은 레포 루트 `render.yaml`(Blueprint)에 있다.

## 한 번만 하는 셋업 (사용자가 Render에서 직접)
1. https://render.com 가입 → **GitHub 연결**(Authorize).
2. **New → Blueprint** → 이 레포(`SEONCHOE/Acupoint_project`) 선택 → Render가 `render.yaml`을 읽어 `acupoint-webapp` 서비스를 자동 생성.
3. 생성 중 **Environment → `OPENAI_API_KEY`** 값 입력(코드/레포엔 저장 안 함).
4. **Create** → 빌드·배포 → `https://acupoint-webapp-XXXX.onrender.com` 발급.

## 동작
- `master`에 push할 때마다 **자동 재배포**(`autoDeploy: true`).
- 무료 플랜은 **15분 무활동 시 슬립** → 첫 요청 콜드스타트 ~30~60초(이후 정상).
- 키 누락 시 자동으로 mock(무과금) 동작 — 실제 추천/STT는 키 필요.

## 보안 / 비용 (꼭)
- `OPENAI_API_KEY`는 **Render 대시보드 환경변수에만**. `.env.local`은 gitignore됨(레포에 없음).
- 공개 URL은 누구나 `/recommend`·`/transcribe` 호출 → OpenAI 요금 소모.
  - **OpenAI 대시보드에서 월 지출 한도(hard limit)** 설정(예: $5) — 최후의 안전장치.
  - 공개 모집이면 접근 비밀번호 / rate-limit 추가 고려.

## 주요 설정 (render.yaml)
- rootDir `acupoint_webapp`, build `pip install -r requirements.txt`,
  start `uvicorn server:app --host 0.0.0.0 --port $PORT`, health `/health`.
- 런타임은 `symptom_match`의 pandas/openpyxl 불필요(json/openai만 사용) → 빠른 빌드.
