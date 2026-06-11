"""
M2(StandardScaler + MLPRegressor) 가중치를 브라우저용 JSON으로 추출.
브라우저 app.js가 onnxruntime 없이 순수 JS forward로 '랜드마크 -> 혈자리'를 추론한다.

  python3 export_m2_weights.py
출력: static/m2_weights.json
"""
import os, json
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.join(HERE, "..", "CV_DeepLearning", "acupoint_api", "models", "m2_model.joblib")
OUT = os.path.join(HERE, "static", "m2_weights.json")

# 혈자리 메타 (core.py ACUP_META 와 동일하게 유지)
ACUP_META = {
    "ekmoon": ("액문", "TE2"), "gwanchung": ("관충", "TE1"), "hapgok": ("합곡", "LI4"),
    "hugye": ("후계", "SI3"), "jungjer": ("중저", "TE3"), "nogung": ("노궁", "PC8"),
    "sangyang": ("상양", "LI1"), "sochung": ("소충", "HT9"), "sotack": ("소택", "SI1"),
    "taeyeon": ("태연", "LU9"), "urjae": ("어제", "LU10"),
}

bundle = joblib.load(M2)
acups = bundle["acups"]
pipe = bundle["model"]
scaler = pipe.steps[0][1]
mlp = pipe.steps[1][1]

acts = ["relu"] * (len(mlp.coefs_) - 1) + [mlp.out_activation_]  # 마지막은 identity
layers = [{"W": W.tolist(), "b": b.tolist(), "act": a}
          for W, b, a in zip(mlp.coefs_, mlp.intercepts_, acts)]

out = {
    "acups": acups,
    "meta": {a: {"name_kr": ACUP_META[a][0], "code": ACUP_META[a][1]} for a in acups},
    "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    "layers": layers,
    "feature_order": "canon(21x2=42, l0x,l0y,...,l20y) + onehot(11)",
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f)
sz = os.path.getsize(OUT) / 1024
print(f"저장: {OUT}  ({sz:.0f} KB) | 혈자리 {len(acups)} | 레이어 {[l['act'] for l in layers]}")
