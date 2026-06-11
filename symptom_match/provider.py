"""
LLM 제공자 추상화 — 앱 런타임 모델을 한 줄로 교체 가능하게.
기본 구현: GPT-4o-mini (OpenAI). Mock은 API 없이 하니스 테스트용.

보안: API 키는 절대 하드코딩하지 않고 환경변수(OPENAI_API_KEY)에서만 읽음.
      누락 시 명확한 에러로 시작 거부.
"""
import os
import json
from abc import ABC, abstractmethod

# LLM이 반환해야 하는 구조 (혈자리는 코드가 KB에서 결정 -> 환각 차단)
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_symptoms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "허용된 증상 목록 중에서만 선택",
        },
        "red_flag": {"type": "boolean", "description": "급성/중증 응급 징후 여부"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "emergency"]},
        "reasoning": {"type": "string", "description": "간단한 판단 근거(한국어)"},
    },
    "required": ["matched_symptoms", "red_flag", "severity", "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """당신은 한의학 자가지압 도우미의 증상 분석기다. 사용자의 한국어 증상 서술을 읽고:
1) 아래 '허용된 증상' 목록에서 해당하는 증상만 고른다. 목록에 없으면 비운다.
2) 급성·중증 응급 징후가 있으면 red_flag=true, severity="emergency"로 표시한다.
   응급 예: 갑작스런 심한 흉통/식은땀, 한쪽 마비·언어장애(뇌졸중), 의식소실, 심한 호흡곤란,
   벼락두통, 고열+목경직+의식저하, 급성 안면마비. 증상명이 목록에 있어도 급성·중증이면 응급으로 본다.
3) 절대 혈자리를 직접 만들어내지 않는다. 증상 분류와 응급 판단만 한다.
반드시 지정된 JSON 스키마로만 답한다."""


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def classify(self, user_text: str, allowed_symptoms: list[str]) -> dict:
        """반환: {matched_symptoms, red_flag, severity, reasoning}"""
        ...


class GPT4oMiniProvider(LLMProvider):
    name = "gpt-4o-mini"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "환경변수 OPENAI_API_KEY 가 없습니다. .env.local 또는 환경변수로 설정하세요. "
                "(키를 코드에 하드코딩하지 마세요.)")
        self.client = OpenAI(api_key=key)
        self.model = model

    def classify(self, user_text, allowed_symptoms):
        sys = SYSTEM_PROMPT + "\n\n허용된 증상: " + ", ".join(allowed_symptoms)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user_text}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "symptom_classification",
                                "schema": CLASSIFY_SCHEMA, "strict": True},
            },
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        # 허용 목록 밖 증상은 폐기(그라운딩 보강)
        allow = set(allowed_symptoms)
        data["matched_symptoms"] = [s for s in data.get("matched_symptoms", []) if s in allow]
        return data


class MockProvider(LLMProvider):
    """API 없이 하니스/플로우 검증용. KB 구어체 키워드 + 응급 키워드 규칙."""
    name = "mock"

    def __init__(self, kb):
        self.kb = kb
        self._emerg = ["갑자기", "쥐어짜", "식은땀", "마비", "어눌", "의식", "쓰러",
                       "숨이", "호흡곤란", "망치", "벼락", "목이 뻣뻣", "돌아가", "뻗치"]

    def classify(self, user_text, allowed_symptoms):
        t = user_text.replace(" ", "")
        red = sum(k.replace(" ", "") in t for k in self._emerg) >= 1 and \
            any(k in user_text for k in ["갑자기", "의식", "마비", "숨이", "망치", "뻣뻣", "뻗치", "어눌"])
        matched = []
        for s in allowed_symptoms:
            cand = [s] + self.kb["colloquial"].get(s, [])
            if any(c.replace(" ", "") in t for c in cand if c):
                matched.append(s)
        return {"matched_symptoms": [] if red else matched,
                "red_flag": bool(red),
                "severity": "emergency" if red else "low",
                "reasoning": "mock 규칙 기반"}
