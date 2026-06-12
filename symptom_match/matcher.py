"""
증상 -> 혈자리 매처. LLM은 증상분류·응급판단만, 혈자리는 KB에서 결정(환각 차단).

  matcher = SymptomMatcher(provider)
  result = matcher.match("머리가 지끈거리고 속이 메스꺼워요")
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))

DISCLAIMER = "본 안내는 의료행위가 아닌 자가관리 참고용입니다. 증상이 심하거나 지속되면 의료기관을 방문하세요."
EMERGENCY_MSG = ("응급 가능성이 있는 증상입니다. 자가지압 대신 즉시 119 또는 가까운 응급실/의료기관에 "
                 "연락하세요.")


class SymptomMatcher:
    def __init__(self, provider, kb_path=os.path.join(HERE, "kb.json")):
        self.kb = json.load(open(kb_path, encoding="utf-8"))
        self.provider = provider
        self.cv = set(self.kb["cv_acupoints"])
        # 주치(主治) 역인덱스: 혈자리 코드 -> 이 혈자리가 다루는 증상들(KB 매핑 기반)
        self.treats: dict[str, list] = {}
        for sym, acs in self.kb["symptom_to_acupoints"].items():
            for a in acs:
                lst = self.treats.setdefault(a["code"], [])
                if sym not in lst:
                    lst.append(sym)

    def match(self, user_text: str) -> dict:
        cls = self.provider.classify(user_text, self.kb["symptoms"])

        if cls.get("red_flag"):
            return {
                "red_flag": True, "severity": "emergency",
                "symptoms": [], "acupoints": [],
                "advice": EMERGENCY_MSG, "disclaimer": DISCLAIMER,
                "reasoning": cls.get("reasoning", ""),
                "provider": self.provider.name,
            }

        # KB에서 혈자리 결정(그라운딩) — 증상별 혈자리 합집합, 코드 기준 dedup
        seen, acupoints = set(), []
        for s in cls.get("matched_symptoms", []):
            for a in self.kb["symptom_to_acupoints"].get(s, []):
                if a["code"] in seen:
                    continue
                seen.add(a["code"])
                acupoints.append({**a, "has_cv_model": a["code"] in self.cv,
                                  "for_symptom": s,
                                  "treats": self.treats.get(a["code"], [])})
        # 손 이미지 오버레이(②) 가능한 혈자리를 앞으로
        acupoints.sort(key=lambda a: (not a["has_cv_model"], a["code"]))

        return {
            "red_flag": False,
            "severity": cls.get("severity", "low"),
            "symptoms": cls.get("matched_symptoms", []),
            "acupoints": acupoints,
            "advice": ("추천 혈자리를 부드럽게 3~5초씩 눌러 자극해 보세요."
                       if acupoints else "해당하는 혈자리를 찾지 못했습니다. 증상을 더 구체적으로 알려주세요."),
            "disclaimer": DISCLAIMER,
            "reasoning": cls.get("reasoning", ""),
            "provider": self.provider.name,
        }
