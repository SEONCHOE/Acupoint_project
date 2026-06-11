#!/usr/bin/env bash
# 혈자리 추론 API 서버 실행.
# MediaPipe는 OpenGL ES 라이브러리(libgles2, libegl1, libglib2.0-0)를 요구한다.
# 이 PC에는 이미 시스템 설치돼 있어 별도 설정 불필요.
# (미설치 환경이면: sudo apt install -y libgles2 libegl1 libglib2.0-0)
cd "$(dirname "$0")"
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 "$@"
