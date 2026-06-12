"""
통합 웹앱 백엔드 (1b) — FastAPI.

역할:
  - POST /recommend : 증상 텍스트 -> ① LLM 증상분석 + KB 혈자리(환각 차단) + 응급분기
  - 정적 프런트(static/) 서빙. CV(랜드마크->혈자리)는 브라우저에서 온디바이스로 수행.

보안: OPENAI_API_KEY 는 환경변수/.env.local 에서만 읽음. 없거나 mock=true 면 규칙기반 Mock 사용.
실행:  uvicorn server:app --port 8000   (또는 ./run_web.sh)
"""
import os
import sys

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
SYMPTOM_DIR = os.path.join(HERE, "..", "symptom_match")
STATIC = os.path.join(HERE, "static")
sys.path.insert(0, SYMPTOM_DIR)

# .env.local(symptom_match) 자동 로드 — 키 하드코딩 금지, 파일에서만 주입
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(SYMPTOM_DIR, ".env.local"))
except Exception:
    pass

import json
from matcher import SymptomMatcher          # noqa: E402  (symptom_match)
from provider import MockProvider           # noqa: E402

app = FastAPI(title="Acupoint Web (1b)")

_KB = json.load(open(os.path.join(SYMPTOM_DIR, "kb.json"), encoding="utf-8"))
_matchers: dict[str, SymptomMatcher] = {}

# STT 모델 — 키 누락 시 호출 안 함(명확한 안내 반환). 코드에 키 하드코딩 금지.
STT_MODEL = os.environ.get("STT_MODEL", "gpt-4o-mini-transcribe")
_oai_client = None


def _openai():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        _oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _oai_client


def get_matcher(use_mock: bool) -> SymptomMatcher:
    key = "mock" if use_mock else "llm"
    if key not in _matchers:
        if use_mock:
            _matchers[key] = SymptomMatcher(MockProvider(_KB))
        else:
            from provider import GPT4oMiniProvider   # 키 없으면 여기서 명확히 에러
            _matchers[key] = SymptomMatcher(GPT4oMiniProvider())
    return _matchers[key]


class RecommendReq(BaseModel):
    text: str
    mock: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
            "n_symptoms": len(_KB["symptoms"]), "cv_acupoints": _KB["cv_acupoints"]}


@app.post("/recommend")
def recommend(req: RecommendReq):
    text = (req.text or "").strip()
    if not text:
        return {"error": "증상을 입력하세요."}
    use_mock = req.mock or not os.environ.get("OPENAI_API_KEY")
    try:
        result = get_matcher(use_mock).match(text)
    except Exception as e:                       # 키 누락·API 오류 등
        return {"error": f"분석 실패: {e}", "fallback_mock": True}
    result["used_mock"] = use_mock
    return result


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """음성(webm/mp4/ogg…) -> 한국어 텍스트. 브라우저 MediaRecorder 업로드를 OpenAI로 전사.
    키 없으면 호출하지 않고 안내만 반환(증상은 텍스트로 입력 가능)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {"text": "", "error": "음성 인식은 OPENAI_API_KEY 가 필요합니다. 텍스트로 입력해 주세요."}
    data = await audio.read()
    if not data:
        return {"text": "", "error": "녹음된 음성이 비어 있습니다."}
    fname = audio.filename or "voice.webm"
    try:
        resp = _openai().audio.transcriptions.create(
            model=STT_MODEL,
            file=(fname, data, audio.content_type or "audio/webm"),
            language="ko",
        )
        return {"text": (resp.text or "").strip(), "model": STT_MODEL}
    except Exception as e:                       # API 오류·포맷 미지원 등
        return {"text": "", "error": f"음성 인식 실패: {e}"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/art")
def index_art():
    """아티스틱 에디션(經穴 · 먹 & 주사). 세련된 기본 버전과 app.js·백엔드 공유."""
    return FileResponse(os.path.join(STATIC, "index-art.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")
