"""
MediaPipe 타당성 스파이크 (혈자리 모델 ②-B 전략 검증).

검증 질문:
  1. 기존 손 이미지에서 MediaPipe Hands가 21 랜드마크를 안정적으로 검출하는가? (검출 성공률)
  2. 정답 혈자리 좌표가 손 랜드마크에 대해 '일관된' 위치에 있는가?
     -> 손 랜드마크로 정의한 정규화 좌표계에서 혈자리 좌표의 분산(std)이 작으면
        "랜드마크 -> 혈자리 오프셋" 학습/규칙 접근(B)이 유효하다는 강한 신호.

산출물:
  - 콘솔: 검출 성공률, 정규화 혈자리 좌표 평균/표준편차
  - overlays/: 랜드마크(흰 점/뼈대) + 정답 혈자리(빨강 X) 오버레이 PNG
"""
import os
import sys
import json
import random
import glob
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----- 설정 -----
ACUP = "hapgok"
BASE = "/mnt/d/Team Project/Acupoint/Hands_processed/Hands_processed/hapgok/hapgok"
JSON_PATH = os.path.join(BASE, f"{ACUP}_info_수정.json")
SIDES = ["hapgok_dorsal_left", "hapgok_dorsal_right"]
N_PER_SIDE = 150        # 검출률 추정용 샘플 수
N_OVERLAYS = 30         # 저장할 오버레이 이미지 수
OUT_DIR = os.path.join(os.path.dirname(__file__), "overlays")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
SEED = 0

# MediaPipe 손 랜드마크 인덱스
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17

# 21 랜드마크 뼈대 연결 (표준 MediaPipe Hands 토폴로지)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 소지
    (0, 17),                                 # 손바닥 밑변
]

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, encoding="utf-8") as f:
    info = json.load(f)


def json_key_for(filename):
    # Hand_0000021.png -> hapgok_0000021
    num = os.path.splitext(filename)[0].split("_")[1]
    return f"{ACUP}_{num}"


def affine_normalize(acup_xy, lms):
    """랜드마크 삼각형(wrist, index_mcp, pinky_mcp)을 캔버스 기저로 삼아
    혈자리 좌표를 affine(barycentric) 좌표로 변환. (스케일/회전/위치 불변)"""
    o = lms[WRIST]
    a = lms[INDEX_MCP] - o
    b = lms[PINKY_MCP] - o
    M = np.stack([a, b], axis=1)          # 2x2
    try:
        uv = np.linalg.solve(M, np.asarray(acup_xy, float) - o)
    except np.linalg.LinAlgError:
        return None
    return uv  # acup = o + u*a + v*b


# ----- 샘플 수집 -----
samples = []
for side in SIDES:
    files = sorted(glob.glob(os.path.join(BASE, side, "*.png")))
    random.shuffle(files)
    for fp in files[:N_PER_SIDE]:
        key = json_key_for(os.path.basename(fp))
        if key in info:
            coord = info[key][1]["acup_coord"]
            samples.append((fp, side, coord))

print(f"[info] 수집 샘플: {len(samples)}장 ({', '.join(SIDES)})")

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

n_total = 0
n_detected = 0
norm_coords = []
saved = 0

for fp, side, coord in samples:
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is None:
        continue
    n_total += 1
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_image)

    if not res.hand_landmarks:
        continue
    n_detected += 1

    h, w = img.shape[:2]
    lm = res.hand_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm])  # 21x2 픽셀

    uv = affine_normalize(coord, pts)
    if uv is not None:
        norm_coords.append(uv)

    if saved < N_OVERLAYS:
        vis = img.copy()
        # 뼈대
        for a, b in HAND_CONNECTIONS:
            pa, pb = pts[a].astype(int), pts[b].astype(int)
            cv2.line(vis, tuple(pa), tuple(pb), (0, 255, 0), 1)
        # 랜드마크 점
        for (x, y) in pts.astype(int):
            cv2.circle(vis, (x, y), 3, (255, 255, 255), -1)
        # 정답 혈자리 (빨강 X)
        cx, cy = int(coord[0]), int(coord[1])
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
        out = os.path.join(OUT_DIR, f"{side}_{os.path.basename(fp)}")
        cv2.imwrite(out, vis)
        saved += 1

detector.close()

# ----- 결과 -----
print("\n========== 결과 ==========")
rate = 100.0 * n_detected / max(n_total, 1)
print(f"검출 성공률: {n_detected}/{n_total} = {rate:.1f}%")

if norm_coords:
    arr = np.array(norm_coords)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print(f"정규화 혈자리 좌표 (affine u,v) 평균: ({mean[0]:.3f}, {mean[1]:.3f})")
    print(f"정규화 혈자리 좌표 표준편차       : ({std[0]:.3f}, {std[1]:.3f})")
    print("  -> std가 작을수록(예: <0.10) 랜드마크 대비 위치가 일관적 = B 전략 유효")
print(f"\n오버레이 {saved}장 저장: {OUT_DIR}")
