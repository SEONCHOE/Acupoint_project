"""
B: 추론 데모 — 임의 손 사진 -> MediaPipe 21랜드마크 -> M2 -> 11혈 좌표 오버레이.

사용:
  python3 demo_predict.py --image path/to/hand.jpg          # 임의 사진에 11혈 예측
  python3 demo_predict.py                                   # 데이터셋 샘플로 예측 vs 정답(GT) 검증

출력: out/demo/*.png  (예측=원, GT=빨강 X)
"""
import os, argparse, zipfile, random
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "hand_landmarker.task")
M2_PATH = os.path.join(HERE, "out", "m2_model.joblib")
OUT = os.path.join(HERE, "out", "demo"); os.makedirs(OUT, exist_ok=True)
ZIP_DIR = "/mnt/d/Team Project/Acupoint/Hands_processed/Hands_processed"
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17

# 혈자리 표시 라벨(로마자+경혈코드)과 색 (BGR)
LABELS = {"ekmoon": "Ekmun TE2", "gwanchung": "Gwanchung TE1", "hapgok": "Hapgok LI4",
          "hugye": "Hugye SI3", "jungjer": "Jungjeo TE3", "nogung": "Nogung PC8",
          "sangyang": "Sangyang LI1", "sochung": "Sochung HT9", "sotack": "Sotaek SI1",
          "taeyeon": "Taeyeon LU9", "urjae": "Eoje LU10"}
def color_for(i, n):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(i / n, 0.9, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def canon_basis(lm):
    o = lm[WRIST]; r = lm[MIDDLE_MCP] - o; s = np.linalg.norm(r)
    s = max(s, 1e-6); e1 = r / s; e2 = np.array([-e1[1], e1[0]])
    vi = lm[INDEX_MCP] - o; vp = lm[PINKY_MCP] - o
    if vi[0] * vp[1] - vi[1] * vp[0] < 0:
        e2 = -e2
    return o, s, e1, e2


def predict_all(lm_px, m2, acups):
    """lm_px:(21,2) -> {acup:(x,y) px}."""
    o, s, e1, e2 = canon_basis(lm_px)
    canon = np.stack([[(p - o) @ e1 / s, (p - o) @ e2 / s] for p in lm_px])  # (21,2)
    feat_lm = canon.reshape(-1)
    X = np.stack([np.concatenate([feat_lm, np.eye(len(acups))[i]]) for i in range(len(acups))])
    pred = m2.predict(X)                       # (n_acup,2) canonical
    out = {}
    for i, a in enumerate(acups):
        px = o + s * (pred[i, 0] * e1 + pred[i, 1] * e2)
        out[a] = px
    return out


def get_landmarks(detector, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])


def draw(img, preds, acups, gt=None, gt_acup=None):
    vis = img.copy()
    for i, a in enumerate(acups):
        x, y = preds[a].astype(int)
        c = color_for(i, len(acups))
        cv2.circle(vis, (x, y), 6, c, -1)
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
        cv2.putText(vis, LABELS[a].split()[0], (x + 7, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    if gt is not None:
        cv2.drawMarker(vis, (int(gt[0]), int(gt[1])), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 22, 3)
        cv2.putText(vis, f"GT:{gt_acup}", (int(gt[0]) + 8, int(gt[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--n", type=int, default=6, help="데이터셋 검증 샘플 수")
    args = ap.parse_args()

    bundle = joblib.load(M2_PATH); m2 = bundle["model"]; acups = bundle["acups"]
    det = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.3))

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("이미지 못 읽음:", args.image); return
        lm = get_landmarks(det, img)
        if lm is None:
            print("손 미검출"); return
        preds = predict_all(lm, m2, acups)
        out = os.path.join(OUT, "pred_" + os.path.basename(args.image))
        cv2.imwrite(out, draw(img, preds, acups))
        print("저장:", out)
        for a in acups:
            print(f"  {LABELS[a]:16} -> ({preds[a][0]:.0f}, {preds[a][1]:.0f})")
        return

    # 데이터셋 검증: 혈자리별 1장씩, 예측 vs GT
    import json
    random.seed(7)
    saved = 0
    for a in acups[:args.n]:
        zp = os.path.join(ZIP_DIR, f"{a}.zip")
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as zf:
            jn = next(n for n in zf.namelist() if n.endswith(".json"))
            info = json.loads(zf.read(jn).decode("utf-8"))
            pngs = [n for n in zf.namelist() if n.endswith(".png") and "dorsal_left" in n]
            random.shuffle(pngs)
            for n in pngs[:1]:
                img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
                lm = get_landmarks(det, img)
                if lm is None:
                    continue
                preds = predict_all(lm, m2, acups)
                key = f"{a}_{os.path.basename(n).split('_')[1].split('.')[0]}"
                gt = info.get(key, [None, {}])[1].get("acup_coord") if key in info else None
                out = os.path.join(OUT, f"verify_{a}.png")
                cv2.imwrite(out, draw(img, preds, acups, gt=gt, gt_acup=a))
                err = np.linalg.norm(preds[a] - np.array(gt)) if gt else float("nan")
                print(f"{a:10} 저장 {os.path.basename(out)}  (해당혈 예측오차 {err:.1f}px)")
                saved += 1
    print(f"\n검증 이미지 {saved}장: {OUT}")


if __name__ == "__main__":
    main()
