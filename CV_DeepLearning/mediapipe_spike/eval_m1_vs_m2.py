"""
M1(현행 ResNet 직접 좌표회귀) vs M2(랜드마크 MLP) 동일 조건 비교.

공정성:
  - M2의 test 분할(seed 0)을 그대로 재현 -> 동일 이미지에서 두 모델 평가.
  - 동일 지표(NME = px오차/손크기, PCK@α). 손크기 s는 landmarks.csv 랜드마크에서 계산.
  - 주의: ResNet 체크포인트의 학습 분할은 미상 -> 이 test 이미지를 학습에 봤을 수 있음(ResNet에 유리).
    그럼에도 M2가 이기면 결론은 보수적으로 확실.
  - 체크포인트 존재 혈자리만: hapgok, ekmoon.

실행: python3 eval_m1_vs_m2.py
"""
import os, io, zipfile
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms
from PIL import Image
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", "landmarks.csv")
M2_PATH = os.path.join(HERE, "out", "m2_model.joblib")
CKPT_DIR = os.path.abspath(os.path.join(HERE, "..", "checkpoints"))
ZIP_DIR = "/mnt/d/Team Project/Acupoint/Hands_processed/Hands_processed"
CKPTS = {
    "hapgok": "hapgok0827_0729org+rot+fill+rotfill_model_best.pth.tar",
    "ekmoon": "ekmoon0828_1500org+rot+fill+rotfill_model_best.pth.tar",
}
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
CANON_CLIP, SEED = 3.0, 0


# ---- MyResNet (resnet18, model_utils.py와 동일) ----
class MyResNet(ResNet):
    def __init__(self, block=BasicBlock, cfg=(2, 2, 2, 2)):
        super().__init__(block, list(cfg))
        self.Linear1 = nn.Linear(512, 1); self.Linear2 = nn.Linear(512, 2)
        self.Linear3 = nn.Linear(512 * 4, 1); self.Linear4 = nn.Linear(512 * 4, 2)
        self.block = block

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.Linear1(x), self.Linear2(x)


def clear_background(img):  # webapp/Image_Process_utils와 동일
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (358, 45, 255))
    mask = 255 - mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    res = img.copy(); res[mask == 0] = (255, 255, 255)
    return res


# ---- canonical (train_m2.py와 동일) ----
def canonical_basis(lm):
    o = lm[:, WRIST]; r = lm[:, MIDDLE_MCP] - o
    s = np.linalg.norm(r, axis=1); ss = np.where(s < 1e-6, 1e-6, s)
    e1 = r / ss[:, None]; e2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)
    vi = lm[:, INDEX_MCP] - o; vp = lm[:, PINKY_MCP] - o
    flip = (vi[:, 0] * vp[:, 1] - vi[:, 1] * vp[:, 0]) < 0
    e2[flip] = -e2[flip]
    return o, ss, e1, e2

def to_canon(p, o, s, e1, e2):
    d = p - o
    return np.stack([np.einsum("ni,ni->n", d, e1) / s,
                     np.einsum("ni,ni->n", d, e2) / s], axis=1)

def from_canon(c, o, s, e1, e2):
    return o + s[:, None] * (c[:, 0:1] * e1 + c[:, 1:2] * e2)


# ---------- M2 test 분할 재현 ----------
df = pd.read_csv(CSV)
df = df[df["detected"] == 1].copy()
acups = sorted(df["acup"].unique()); aidx = {a: i for i, a in enumerate(acups)}
ids = df["img_id"].values; hp = df["hand_pos"].values; ac = df["acup"].values
w = df["w"].values.astype(float); h = df["h"].values.astype(float)
xy = np.empty((len(df), 21, 2))
for i in range(21):
    xy[:, i, 0] = df[f"x{i}"].values * w; xy[:, i, 1] = df[f"y{i}"].values * h
acup_px = df[["acup_x", "acup_y"]].values.astype(float)
o, s, e1, e2 = canonical_basis(xy)
lm_canon = np.stack([to_canon(xy[:, i], o, s, e1, e2) for i in range(21)], axis=1)
y_canon = to_canon(acup_px, o, s, e1, e2)
ok = (np.abs(lm_canon).max(axis=(1, 2)) < CANON_CLIP) & (np.abs(y_canon).max(axis=1) < CANON_CLIP)
sel = np.where(ok)[0]
rng = np.random.default_rng(SEED); order = np.arange(len(sel)); rng.shuffle(order)
n_test = int(0.10 * len(sel)); test_local = order[:n_test]
test_rows = sel[test_local]            # 원본(detected) df 위치
print(f"M2 test 재현: {len(test_rows):,}행 (전체 detected {len(df):,})")

# ---------- M2 예측 (동일 test 행) ----------
bundle = joblib.load(M2_PATH); m2 = bundle["model"]
onehot = np.eye(len(acups))[[aidx[a] for a in ac[test_rows]]]
X = np.concatenate([lm_canon[test_rows].reshape(len(test_rows), -1), onehot], axis=1)
m2_canon = m2.predict(X)
m2_px = from_canon(m2_canon, o[test_rows], s[test_rows], e1[test_rows], e2[test_rows])

# ---------- ResNet(M1) 예측 ----------
to_tensor = transforms.ToTensor()
zips = {a: zipfile.ZipFile(os.path.join(ZIP_DIR, f"{a}.zip")) for a in CKPTS}
models = {}
for a, fn in CKPTS.items():
    ck = torch.load(os.path.join(CKPT_DIR, fn), map_location="cpu", weights_only=False)
    m = MyResNet(); m.load_state_dict(ck["state_dict"]); m.eval()
    models[a] = m

def resnet_pred_px(acup, img_id, hand_pos, w_):
    num = img_id.split("_")[1]
    path = f"{acup}/{acup}_{hand_pos}/Hand_{num}.png"
    try:
        data = zips[acup].read(path)
    except KeyError:
        return None
    arr = np.frombuffer(data, np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(clear_background(img), cv2.COLOR_BGR2RGB)
    t = to_tensor(Image.fromarray(img)).unsqueeze(0)
    with torch.no_grad():
        _, coord = models[acup](t)
    x256, y256 = coord.numpy().squeeze()
    return np.array([x256 * w_ / 256.0, y256 * w_ / 256.0])


# ---------- 비교 (hapgok, ekmoon) ----------
def metrics(err, s_):
    nme = err / s_
    return (nme.mean(), np.median(err),
            100 * (nme < 0.05).mean(), 100 * (nme < 0.10).mean(), 100 * (nme < 0.15).mean())

print(f"\n{'혈자리':9} {'모델':6} {'n':>4} | {'NME':>6} {'px중앙':>6} | {'@0.05':>6} {'@0.10':>6} {'@0.15':>6}")
print("-" * 70)
for a in CKPTS:
    mrows = np.where(ac[test_rows] == a)[0]
    if len(mrows) == 0:
        continue
    gt = acup_px[test_rows][mrows]; s_ = s[test_rows][mrows]
    # M2
    e_m2 = np.linalg.norm(m2_px[mrows] - gt, axis=1)
    nme, med, p5, p10, p15 = metrics(e_m2, s_)
    print(f"{a:9} {'M2':6} {len(mrows):>4} | {nme:>6.4f} {med:>6.1f} | {p5:>5.1f}% {p10:>5.1f}% {p15:>5.1f}%")
    # M1
    preds = []
    valid = []
    for j in mrows:
        p = resnet_pred_px(a, ids[test_rows][j], hp[test_rows][j], w[test_rows][j])
        if p is not None:
            preds.append(p); valid.append(j)
    preds = np.array(preds); valid = np.array(valid)
    gt1 = acup_px[test_rows][valid]; s1 = s[test_rows][valid]
    e_m1 = np.linalg.norm(preds - gt1, axis=1)
    nme, med, p5, p10, p15 = metrics(e_m1, s1)
    print(f"{a:9} {'M1(RN)':6} {len(valid):>4} | {nme:>6.4f} {med:>6.1f} | {p5:>5.1f}% {p10:>5.1f}% {p15:>5.1f}%")
    print("-" * 70)
