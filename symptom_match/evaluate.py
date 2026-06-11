"""
골든 평가셋으로 매처(=특정 LLM) 성능 채점. 모델 선택의 '자'.

  python3 evaluate.py --provider mock          # API 없이 하니스 검증
  python3 evaluate.py --provider gpt-4o-mini    # 실제 API(과금) — OPENAI_API_KEY 필요

지표:
  - 증상 F1 (비응급)        : 분류 정확도
  - 혈자리 recall (비응급)  : 기대 혈자리를 얼마나 포함
  - 응급 recall            : 응급을 놓치지 않는가 (가장 중요, 안전)
  - 비응급 오탐율          : 정상/경증을 응급으로 잘못 보는 비율
"""
import os, json, argparse
from matcher import SymptomMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_eval.jsonl")


def prf(pred:set, gold:set):
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def make_provider(name):
    kb = json.load(open(os.path.join(HERE, "kb.json"), encoding="utf-8"))
    if name == "mock":
        from provider import MockProvider
        return MockProvider(kb)
    if name in ("gpt-4o-mini", "gpt4o-mini"):
        from provider import GPT4oMiniProvider
        return GPT4oMiniProvider()
    raise SystemExit(f"알 수 없는 provider: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    matcher = SymptomMatcher(make_provider(args.provider))
    items = [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]
    if args.limit:
        items = items[:args.limit]

    symp_f, acu_r = [], []
    emerg_total = emerg_hit = 0
    nonemerg_total = nonemerg_fp = 0
    errors = []

    for it in items:
        res = matcher.match(it["input"])
        gold_red = it["red_flag"]
        pred_red = res["red_flag"]

        if gold_red:
            emerg_total += 1
            emerg_hit += int(pred_red)
            if not pred_red:
                errors.append(("응급놓침", it["input"]))
        else:
            nonemerg_total += 1
            if pred_red:
                nonemerg_fp += 1
                errors.append(("응급오탐", it["input"]))
            else:
                _, _, f = prf(set(res["symptoms"]), set(it["expected_symptoms"]))
                symp_f.append(f)
                _, r, _ = prf({a["code"] for a in res["acupoints"]},
                              set(it["expected_acupoint_codes"]))
                acu_r.append(r)

    def avg(x): return sum(x) / len(x) if x else float("nan")
    print(f"\n===== 평가: provider={args.provider}, {len(items)}건 =====")
    print(f"증상 F1 (비응급)     : {avg(symp_f):.3f}")
    print(f"혈자리 recall (비응급): {avg(acu_r):.3f}")
    print(f"응급 recall (안전)   : {emerg_hit}/{emerg_total} = "
          f"{100*emerg_hit/emerg_total if emerg_total else 0:.0f}%  <-- 놓치면 안 됨")
    print(f"비응급 오탐율        : {nonemerg_fp}/{nonemerg_total} = "
          f"{100*nonemerg_fp/nonemerg_total if nonemerg_total else 0:.1f}%")
    if errors:
        print("\n오류 사례(최대 8):")
        for kind, txt in errors[:8]:
            print(f"  [{kind}] {txt}")


if __name__ == "__main__":
    main()
