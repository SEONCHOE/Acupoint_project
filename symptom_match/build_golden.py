"""
골든 평가셋 생성 — 성능 테스트/모델 선택용 '고정 시험지'. (파인튜닝 데이터 아님)

구성:
  - colloquial : KB의 실제 구어체 표현 -> 단일 증상 (현실 입력)
  - multi      : 두 증상 결합
  - typo       : 구어체에 오타 주입
  - emergency  : 급성/중증 red-flag (DB에 증상명이 있어도 병원안내가 정답)
  - benign     : 응급과 같은 증상명의 '경증/만성' 버전 (red_flag 과민 방지 대조)

각 항목: {id, input, type, expected_symptoms[], expected_acupoint_codes[], red_flag, severity, note}
산출: golden_eval.jsonl
"""
import os, json, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
KB = json.load(open(os.path.join(HERE, "kb.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "golden_eval.jsonl")
SEED = 7
random.seed(SEED)


# 급성·중증을 가리키는 구어체 표현: 증상명이 KB에 있어도 이 표현은 병원 안내가 정답.
# (쓰러지다=실신, 편마비=중풍/뇌졸중, 가슴통증=흉통/협심증 의심. '다리를 절다' 등 경증
#  구어체는 제외 — 표현 단위로 응급 여부를 가린다.)
EMERGENCY_COLLOQUIAL = {
    "쓰러지다", "편마비", "가슴이 아프다", "가슴이 바늘로 찌르듯 아프다",
}


def codes_for(symptom):
    return [a["code"] for a in KB["symptom_to_acupoints"].get(symptom, [])]


def typo(s):
    # 간단한 오타: 한 글자 자모 흔들기 대신 띄어쓰기 제거/조사 변형
    s2 = s.replace(" ", "")
    s2 = re.sub(r"요$", "용", s2)         # ~해요 -> ~해용
    s2 = re.sub(r"아프다", "아푸다", s2)
    return s2 if s2 != s else s + "ㅠ"


items = []
_id = 0
def add(**kw):
    global _id
    _id += 1
    items.append({"id": f"g{_id:03d}", **kw})


# ---- 1) colloquial: 증상별 구어체 1개 (순환성 과대표집 방지 -> 표본 재균형) ----
for s, exprs in KB["colloquial"].items():
    if not exprs:
        continue
    for e in exprs[:1]:
        if e in EMERGENCY_COLLOQUIAL:
            add(input=e, type="emergency", expected_symptoms=[],
                expected_acupoint_codes=[], red_flag=True, severity="emergency",
                note=f"응급(구어체): {s} 급성 표현 -> 병원 안내")
        else:
            add(input=e, type="colloquial", expected_symptoms=[s],
                expected_acupoint_codes=codes_for(s), red_flag=False,
                severity="low", note="실제 구어체 표현")

# ---- 2) multi: 자주 같이 오는 증상 결합 ----
multi_pairs = [
    ("두통", "오심", "머리가 지끈거리고 속이 메스꺼워요"),
    ("기침", "인후염", "기침이 심하고 목이 따갑고 아파요"),
    ("불면", "두근거림", "잠도 안 오고 가슴이 자꾸 두근거려요"),
    ("수족냉증", "피로", "손발이 차갑고 늘 기운이 없어요"),
    ("소화장애", "복통", "소화가 안 되고 배가 살살 아파요"),
]
for s1, s2, text in multi_pairs:
    add(input=text, type="multi", expected_symptoms=[s1, s2],
        expected_acupoint_codes=sorted(set(codes_for(s1) + codes_for(s2))),
        red_flag=False, severity="low", note="복합 증상")

# ---- 3) typo: 구어체에 오타 ----
typo_src = [("두통", "머리아프다"), ("불면", "잠을 못잔다"),
            ("수족냉증", "손이 차다"), ("기침", "기침이 난다")]
for s, e in typo_src:
    add(input=typo(e), type="typo", expected_symptoms=[s],
        expected_acupoint_codes=codes_for(s), red_flag=False,
        severity="low", note=f"오타 변형(원문:{e})")

# ---- 4) emergency: 급성/중증 -> 병원안내가 정답 (혈자리 제공 아님) ----
emergencies = [
    ("갑자기 가슴이 쥐어짜듯 아프고 식은땀이 나요", "급성 흉통(심근경색 의심)"),
    ("한쪽 팔다리에 갑자기 힘이 빠지고 말이 어눌해졌어요", "뇌졸중 의심"),
    ("갑자기 의식을 잃고 쓰러졌어요", "실신/의식소실"),
    ("숨이 너무 차서 말을 못 할 정도예요", "급성 호흡곤란"),
    ("머리가 망치로 맞은 듯 갑자기 극심하게 아파요", "벼락두통(지주막하출혈 의심)"),
    ("고열이 나면서 목이 뻣뻣하고 정신이 흐려져요", "수막염 의심"),
    ("입이 한쪽으로 돌아가고 눈이 안 감겨요", "급성 안면마비/뇌졸중 의심"),
    ("가슴 통증이 왼팔로 뻗치고 어지러워요", "방사통 동반 흉통"),
]
for text, why in emergencies:
    add(input=text, type="emergency", expected_symptoms=[],
        expected_acupoint_codes=[], red_flag=True, severity="emergency",
        note=f"응급: {why} -> 병원 안내")

# ---- 5) benign 대조: 같은 증상명이지만 경증/만성 (red_flag=False 여야 함) ----
benign = [
    ("며칠 전부터 어깨랑 등이 뻐근하게 결려요", "견배통", "만성 견배통(경증)"),
    ("요즘 머리가 가끔 지끈거려요", "두통", "경증 두통"),
    ("환절기라 그런지 가벼운 기침이 나요", "기침", "경증 기침"),
]
for text, s, why in benign:
    add(input=text, type="benign", expected_symptoms=[s],
        expected_acupoint_codes=codes_for(s), red_flag=False,
        severity="low", note=why)

# ---- 6) paraphrase: KB 구어체 사전에 '없는' 자연어 표현 (키워드 매칭 불가 -> 의미 이해 필요) ----
# 변별력 핵심. 각 문장은 해당 증상의 KB 구어체·정식명칭을 substring으로 포함하지 않음.
paraphrases = [
    ("두통", "관자놀이가 지끈지끈 울려서 도무지 집중이 안 돼요"),
    ("불면", "새벽 세 시까지 눈만 말똥말똥하고 천장만 바라봐요"),
    ("변비", "사흘째 화장실을 못 가서 아랫배가 묵직하고 답답해요"),
    ("설사", "장이 안 좋은지 변이 죽처럼 나오고 배가 자주 부글거려요"),
    ("수족냉증", "한여름에도 손끝 발끝이 시려서 양말을 못 벗어요"),
    ("피로", "푹 쉬어도 몸이 축 처지고 만사가 귀찮아요"),
    ("이명", "조용한 방에 있으면 귀에서 가느다란 금속음이 계속 들려요"),
    ("비염", "환절기만 되면 재채기가 멈추질 않고 코가 간질거려요"),
    ("치통", "찬물을 마실 때마다 어금니가 욱신거려요"),
    ("요통", "무거운 걸 든 뒤로 허리를 펼 때마다 찌릿해요"),
    ("생리통", "그날만 되면 하복부가 비틀리듯 아파서 진통제를 먹어요"),
    ("안구피로", "모니터를 오래 보면 눈이 뻑뻑하고 침침해져요"),
    ("현훈", "피곤하면 주변이 빙 도는 느낌이 들 때가 있어요"),
    ("오심", "차만 타면 속이 뒤집힐 것 같고 신물이 올라와요"),
    ("건망", "어제 점심에 뭘 먹었는지도 가물가물하고 약속을 자꾸 놓쳐요"),
    ("인후염", "목이 칼칼하고 침 삼킬 때 따끔거려요"),
]
for s, text in paraphrases:
    add(input=text, type="paraphrase", expected_symptoms=[s],
        expected_acupoint_codes=codes_for(s), red_flag=False,
        severity="low", note="KB에 없는 패러프레이즈(의미 이해 필요)")

# ---- 7) multi 패러프레이즈: 복합 증상 + KB 키워드 회피 ----
multi_para = [
    ("두통", "불면", "며칠째 잠을 설쳤더니 관자놀이가 조여오고 머리가 멍해요"),
    ("소화장애", "변비", "명치가 답답하고 일주일째 화장실을 못 가서 아랫배가 단단해요"),
    ("수족냉증", "피로", "손끝이 늘 시리고 몸이 축 처져서 자꾸 눕고만 싶어요"),
    ("안구피로", "두통", "하루종일 화면을 봤더니 눈이 시큰하고 뒷골이 당겨요"),
    ("기침", "인후염", "목이 따끔거리면서 캑캑거리고 가래가 끓어요"),
]
for s1, s2, text in multi_para:
    add(input=text, type="multi", expected_symptoms=[s1, s2],
        expected_acupoint_codes=sorted(set(codes_for(s1) + codes_for(s2))),
        red_flag=False, severity="low", note="복합 증상 패러프레이즈(KB 키워드 회피)")

# ---- 8) typo 패러프레이즈: 구어체가 아닌 표현에 실제 오타/구어 변형 ----
typo_para = [
    ("치통", "이가 욱씬거려서 밥을 못 씹겠어여"),
    ("설사", "배탈나서 화장실을 들락날락해써요"),
    ("피로", "요새 너무 피곤해서 암것도 못하겠어염"),
]
for s, text in typo_para:
    add(input=text, type="typo", expected_symptoms=[s],
        expected_acupoint_codes=codes_for(s), red_flag=False,
        severity="low", note="패러프레이즈+오타")

with open(OUT, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

from collections import Counter
c = Counter(it["type"] for it in items)
print(f"골든 평가셋 저장: {OUT}")
print(f"  총 {len(items)}건 | " + ", ".join(f"{k}:{v}" for k, v in c.items()))
print(f"  응급(red_flag) {sum(it['red_flag'] for it in items)}건")
