"""
지식베이스(KB) 빌더 — acupoint_v3 xlsx + Symptom.db 를 RAG/매칭용 kb.json 으로 변환.

KB는 이 시스템의 '정답(ground truth)'. LLM은 자유 생성하지 않고 KB 안에서만 혈자리를 고른다.
산출: kb.json
  - symptoms: 혈자리 매핑이 있는 정규 증상 목록
  - symptom_to_acupoints: {증상: [{acupoint, code, meridian, position}]}
  - colloquial: {증상: [구어체 표현...]}   (st_symptom)
  - category: {증상: 분류}                  (symptom)
  - cv_acupoints: 손 이미지 모델(②)이 보유한 11혈 코드
"""
import os, re, json
from collections import defaultdict
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "Text Searching", "Symptom_Matching", "acupoint_v3_0822.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb.json")

# ②의 손 혈자리 CV 모델이 커버하는 경혈코드 (m2_model)
CV_ACUPOINTS = ["LU9", "LU10", "LI1", "LI4", "SI1", "SI3", "HT9", "PC8", "TE1", "TE2", "TE3"]


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def main():
    xl = pd.ExcelFile(XLSX)
    acu = xl.parse("acu_target")
    symp = xl.parse("symptom")
    st = xl.parse("st_symptom")

    s2a = defaultdict(list)
    for _, r in acu.iterrows():
        meta = {"acupoint": norm(r["Acupoint"]), "code": norm(r["Acu_symbol"]),
                "meridian": norm(r["Meridian"]), "position": norm(r["Acu_position"])}
        for s in str(r["Target"]).split(","):
            s = norm(s)
            if s:
                s2a[s].append(meta)

    colloquial = defaultdict(list)
    for _, r in st.iterrows():
        colloquial[norm(r["Symptom"])].append(norm(r["value"]))

    category = {norm(r["Symptom"]): norm(r["Category"]) for _, r in symp.iterrows()}

    symptoms = sorted(s2a.keys())
    kb = {
        "symptoms": symptoms,
        "symptom_to_acupoints": {s: s2a[s] for s in symptoms},
        "colloquial": {s: colloquial.get(s, []) for s in symptoms},
        "category": {s: category.get(s, "기타") for s in symptoms},
        "cv_acupoints": CV_ACUPOINTS,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"KB 저장: {OUT}")
    print(f"  증상 {len(symptoms)}개, 혈자리 매핑 {sum(len(v) for v in s2a.values())}건")
    print(f"  구어체 표현 {sum(len(v) for v in colloquial.values())}개")


if __name__ == "__main__":
    main()
