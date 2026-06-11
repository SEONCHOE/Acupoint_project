#!/usr/bin/env bash
# 통합 웹앱(1b) 실행. CV는 브라우저에서 돌아 GL 라이브러리 불필요.
# 키는 ../symptom_match/.env.local 의 OPENAI_API_KEY 를 server.py 가 자동 로드(없으면 mock).
cd "$(dirname "$0")"
exec uvicorn server:app --host 0.0.0.0 --port 8000 "$@"
