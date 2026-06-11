"""
앱 런타임 진입점 — 증상 한 줄 -> 혈자리 추천(JSON). 기본 모델: GPT-4o-mini.

  python3 cli.py "머리가 지끈거리고 속이 메스꺼워요"     # gpt-4o-mini (OPENAI_API_KEY 필요)
  python3 cli.py --mock "손발이 차고 기운이 없어요"        # API 없이 규칙 기반
"""
import sys, json, argparse
from matcher import SymptomMatcher


def build_matcher(use_mock: bool):
    if use_mock:
        import json as _j, os
        from provider import MockProvider
        kb = _j.load(open(os.path.join(os.path.dirname(__file__), "kb.json"), encoding="utf-8"))
        return SymptomMatcher(MockProvider(kb))
    from provider import GPT4oMiniProvider   # 기본: 앱 런타임 모델
    return SymptomMatcher(GPT4oMiniProvider())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--mock", action="store_true", help="API 없이 규칙 기반")
    args = ap.parse_args()
    matcher = build_matcher(args.mock)
    result = matcher.match(" ".join(args.text))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
