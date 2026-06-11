# Symptom → Acupoint Matching (①)

사용자의 한국어 증상 서술 → LLM 증상분석(+응급판단) → **KB(정답 DB)에서 혈자리 결정**.
LLM은 혈자리를 자유 생성하지 않는다(환각 차단). 앱 런타임 기본 모델: **GPT-4o-mini**(교체 가능).

## 구성
```
symptom_match/
├── build_kb.py       # acupoint_v3 xlsx + Symptom.db → kb.json (정답 지식베이스)
├── kb.json           # 51증상 / 158 혈자리매핑 / 304 구어체표현 / CV 11혈 표시
├── build_golden.py   # 골든 평가셋 생성기
├── golden_eval.jsonl # 평가셋 104건 (구어체·복합·오타·응급8·경증대조)  ← 성능 테스트용(파인튜닝 X)
├── provider.py       # LLMProvider 인터페이스 + GPT4oMiniProvider + MockProvider
├── matcher.py        # SymptomMatcher: 분류→KB 혈자리 결정→응급분기
├── evaluate.py       # 골든셋 채점 (모델 선택의 '자')
├── cli.py            # 앱 런타임 진입점
├── .env.example      # OPENAI_API_KEY (실제 키는 .env.local, 커밋 금지)
└── requirements.txt
```

## 동작 원리 (안전 설계)
1. LLM은 `{matched_symptoms(허용목록 내), red_flag, severity, reasoning}` 만 반환.
2. **혈자리는 코드가 KB에서 조회** → 환각 불가능.
3. `red_flag=true`(급성·중증)면 혈자리 대신 **병원/119 안내**. 증상명이 DB에 있어도 급성이면 응급 처리.
4. 추천 혈자리 중 손 이미지 모델(②)이 커버하는 11혈은 `has_cv_model:true`로 앞에 배치 → 손 위 오버레이 연계.

## 사용
```bash
# 1) 키 설정 (코드에 하드코딩 금지)
cp .env.example .env.local && echo "OPENAI_API_KEY=sk-..." >> .env.local   # 또는 export OPENAI_API_KEY=...

# 2) 단일 질의 (앱 런타임, GPT-4o-mini)
python3 cli.py "머리가 지끈거리고 속이 메스꺼워요"
python3 cli.py --mock "..."        # API 없이 규칙 기반(개발용)

# 3) 모델 평가 (골든셋)
python3 evaluate.py --provider mock          # 하니스 검증(무과금)
python3 evaluate.py --provider gpt-4o-mini   # 실제 성능(API 과금)
```

## 모델 교체
`provider.py`의 `LLMProvider`를 구현하면 한 줄로 교체. 향후 Claude/Gemini 추가 시
`evaluate.py`로 **같은 골든셋에서 비교** → 안전·품질 통과하는 가장 싼 모델 채택.

## 출력 예
```json
{
  "red_flag": false, "severity": "low",
  "symptoms": ["두통", "오심"],
  "acupoints": [{"acupoint":"중저","code":"TE3","has_cv_model":true,"for_symptom":"두통", ...}],
  "advice": "추천 혈자리를 부드럽게 3~5초씩 눌러 자극해 보세요.",
  "disclaimer": "본 안내는 의료행위가 아닌 자가관리 참고용입니다 ...",
  "provider": "gpt-4o-mini"
}
```

## 평가셋 성격
`golden_eval.jsonl`은 **성능 테스트/모델 선택용 고정 시험지**이지 파인튜닝 데이터가 아니다.
정답 출처는 KB(기존 DB)이며, 응급 8건은 "급성이면 병원안내가 정답"을 검증한다.
```
