"""
M2: 통합 '손 랜드마크 -> 혈자리 좌표' 회귀 모델.

핵심 아이디어
  - 21 랜드마크로 similarity(이동·회전·스케일 + 좌우손 chirality) 정규화 좌표계를 만든다.
  - 혈자리 좌표(픽셀)를 같은 좌표계로 옮겨 학습 타깃으로 쓴다.
  - 단일 MLP가 (정규화 랜드마크 + 혈자리 one-hot) -> 정규화 혈자리 좌표 를 예측.
  - 추론 시 역변환으로 픽셀 좌표 복원. 평가는 픽셀/손크기 기준 PCK·NME.

대조군
  - 상수 예측(혈자리별 학습셋 중앙 canonical 위치) = "고정 오프셋 규칙". MLP가 이걸 넘는지 확인.

실행:  python3 train_m2.py
출력:  out/m2_model.joblib, out/m2_metrics.csv, 콘솔 리포트
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", "landmarks.csv")
OUT_MODEL = os.path.join(HERE, "out", "m2_model.joblib")
OUT_METRICS = os.path.join(HERE, "out", "m2_metrics.csv")
SEED = 0

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
CANON_CLIP = 3.0      # |canonical 좌표|>3 손길이 => 퇴화 검출로 보고 제외
rng = np.random.default_rng(SEED)


def canonical_basis(lm_px):
    """lm_px: (N,21,2) 픽셀. 반환: o(N,2), s(N,), e1(N,2), e2(N,2)."""
    o = lm_px[:, WRIST]                              # (N,2)
    r = lm_px[:, MIDDLE_MCP] - o
    s = np.linalg.norm(r, axis=1)                    # (N,)
    s_safe = np.where(s < 1e-6, 1e-6, s)
    e1 = r / s_safe[:, None]
    e2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)     # perp
    # chirality 정규화: (index-wrist) x (pinky-wrist) 부호로 좌/우손 통일
    vi = lm_px[:, INDEX_MCP] - o
    vp = lm_px[:, PINKY_MCP] - o
    cross = vi[:, 0] * vp[:, 1] - vi[:, 1] * vp[:, 0]
    flip = cross < 0
    e2[flip] = -e2[flip]
    return o, s_safe, e1, e2


def to_canonical(pts, o, s, e1, e2):
    """pts:(N,2) 픽셀 -> canonical. 벡터화."""
    d = pts - o
    c0 = np.einsum("ni,ni->n", d, e1) / s
    c1 = np.einsum("ni,ni->n", d, e2) / s
    return np.stack([c0, c1], axis=1)


def from_canonical(c, o, s, e1, e2):
    """canonical -> 픽셀."""
    return o + s[:, None] * (c[:, 0:1] * e1 + c[:, 1:2] * e2)


# ---------- 데이터 ----------
df = pd.read_csv(CSV)
df = df[df["detected"] == 1].copy()
acups = sorted(df["acup"].unique())
aidx = {a: i for i, a in enumerate(acups)}
print(f"검출 행: {len(df):,} | 혈자리 {len(acups)}개: {acups}")

# 랜드마크 픽셀 좌표 (N,21,2)
xy = np.empty((len(df), 21, 2), dtype=float)
for i in range(21):
    xy[:, i, 0] = df[f"x{i}"].values * df["w"].values
    xy[:, i, 1] = df[f"y{i}"].values * df["h"].values
acup_px = df[["acup_x", "acup_y"]].values.astype(float)

o, s, e1, e2 = canonical_basis(xy)
# 정규화 랜드마크 (N,21,2) & 타깃
lm_canon = np.stack([to_canonical(xy[:, i], o, s, e1, e2) for i in range(21)], axis=1)
y_canon = to_canonical(acup_px, o, s, e1, e2)

# 퇴화 검출 필터 (랜드마크나 타깃이 비정상적으로 멀리)
ok = (np.abs(lm_canon).max(axis=(1, 2)) < CANON_CLIP) & (np.abs(y_canon).max(axis=1) < CANON_CLIP)
print(f"퇴화 검출 제외: {(~ok).sum():,} ({100*(~ok).mean():.1f}%) -> 사용 {ok.sum():,}")
lm_canon, y_canon = lm_canon[ok], y_canon[ok]
o, s, e1, e2 = o[ok], s[ok], e1[ok], e2[ok]
acup_id = df["acup"].map(aidx).values[ok]
acup_px = acup_px[ok]

# 특징: 정규화 랜드마크(42) + 혈자리 one-hot
onehot = np.eye(len(acups))[acup_id]
X = np.concatenate([lm_canon.reshape(len(lm_canon), -1), onehot], axis=1)

# ---------- split (혈자리 층화) ----------
idx = np.arange(len(X))
rng.shuffle(idx)
n_test = int(0.10 * len(idx)); n_val = int(0.10 * len(idx))
test_i, val_i, train_i = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
print(f"split  train {len(train_i):,} | val {len(val_i):,} | test {len(test_i):,}")

# ---------- 학습 ----------
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                 alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                 max_iter=300, early_stopping=True, n_iter_no_change=15,
                 random_state=SEED, verbose=False),
)
print("학습 중 ...", flush=True)
model.fit(X[train_i], y_canon[train_i])


# ---------- 평가 ----------
def pixel_errors(pred_canon, sel):
    pred_px = from_canonical(pred_canon, o[sel], s[sel], e1[sel], e2[sel])
    err = np.linalg.norm(pred_px - acup_px[sel], axis=1)   # 픽셀 오차
    nme = err / s[sel]                                      # 손크기 정규화
    return err, nme


def report(name, sel, pred_canon):
    err, nme = pixel_errors(pred_canon, sel)
    line = (f"{name:16} | NME {nme.mean():.4f} | px median {np.median(err):5.1f} "
            f"| PCK@0.05 {100*(nme<0.05).mean():4.1f}% | @0.10 {100*(nme<0.10).mean():4.1f}% "
            f"| @0.15 {100*(nme<0.15).mean():4.1f}%")
    print(line)
    return err, nme


print("\n===== 테스트 평가 =====")
# 상수 베이스라인: 학습셋 혈자리별 canonical 중앙값
const_map = {}
for a in range(len(acups)):
    m = acup_id[train_i] == a
    const_map[a] = np.median(y_canon[train_i][m], axis=0) if m.any() else np.array([0., 0.])
base_pred = np.stack([const_map[a] for a in acup_id[test_i]])
report("상수규칙(대조)", test_i, base_pred)

mlp_pred = model.predict(X[test_i])
report("MLP(전체)", test_i, mlp_pred)

# 혈자리별 MLP 성능
print("\n----- 혈자리별 (MLP, test) -----")
rows = []
err_all, nme_all = pixel_errors(mlp_pred, test_i)
for a in range(len(acups)):
    m = acup_id[test_i] == a
    if not m.any():
        continue
    nme = nme_all[m]; err = err_all[m]
    pck10 = 100 * (nme < 0.10).mean()
    print(f"{acups[a]:10} n={m.sum():>4} | NME {nme.mean():.4f} | px med {np.median(err):5.1f} | PCK@0.10 {pck10:5.1f}%")
    rows.append([acups[a], int(m.sum()), float(nme.mean()), float(np.median(err)), float(pck10)])

pd.DataFrame(rows, columns=["acup", "n", "nme", "px_median", "pck@0.10"]).to_csv(OUT_METRICS, index=False)
joblib.dump({"model": model, "acups": acups}, OUT_MODEL)
print(f"\n모델 저장: {OUT_MODEL}\n지표 저장: {OUT_METRICS}")
