"""
전 혈자리 배치 추출 — zip에서 직접 읽어(디스크에 안 풂) MediaPipe로 21 랜드마크를 뽑고
'(랜드마크 -> 혈자리 오프셋)' 학습 테이블 + 혈자리별 검출률/일관성 리포트를 생성.

사용:
  python3 extract_all.py --sample 250          # 그룹당 250장 샘플 (빠른 리포트)
  python3 extract_all.py --sample 0            # 전수 처리 (학습셋 풀빌드, 느림)
  python3 extract_all.py --acups hapgok nogung # 특정 혈자리만

산출물 (out/):
  landmarks.csv  : 한 행 = 이미지 1장. meta + acup_coord(px) + 정규화(affine u,v) + 21*(x,y,z) 정규화 랜드마크
  report.csv     : 혈자리×hand_pos 별 검출률, affine u/v 평균·표준편차
"""
import os
import io
import csv
import json
import glob
import argparse
import zipfile
import random
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ZIP_DIR = "/mnt/d/Team Project/Acupoint/Hands_processed/Hands_processed"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
OUT_DIR = os.path.join(HERE, "out")
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
SEED = 0

ALL_ACUPS = ["ekmoon", "gwanchung", "hapgok", "hugye", "jungjer", "nogung",
             "sangyang", "shinmoom", "sochung", "sotack", "taeyeon", "urjae"]


def make_detector():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def affine_uv(acup_xy, pts):
    o = pts[WRIST]
    M = np.stack([pts[INDEX_MCP] - o, pts[PINKY_MCP] - o], axis=1)
    try:
        return np.linalg.solve(M, np.asarray(acup_xy, float) - o)
    except np.linalg.LinAlgError:
        return None


def json_key(acup, filename):
    num = os.path.splitext(os.path.basename(filename))[0].split("_")[1]
    return f"{acup}_{num}"


def load_info(zf):
    jname = next(n for n in zf.namelist() if n.lower().endswith(".json"))
    return json.loads(zf.read(jname).decode("utf-8"))


def process(acups, sample, source):
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    detector = make_detector()

    lm_path = os.path.join(OUT_DIR, "landmarks.csv")
    header = (["acup", "hand_pos", "img_id", "detected", "w", "h",
               "acup_x", "acup_y", "u", "v"]
              + [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")])
    lm_file = open(lm_path, "w", newline="", encoding="utf-8")
    lm_writer = csv.writer(lm_file)
    lm_writer.writerow(header)

    # 집계: (acup, pos) -> [n_total, n_det, list(u), list(v)]
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, [], []])

    for acup in acups:
        zp = os.path.join(ZIP_DIR, f"{acup}.zip")
        if not os.path.exists(zp):
            print(f"[skip] {acup}: zip 없음")
            continue
        with zipfile.ZipFile(zp) as zf:
            info = load_info(zf)
            pngs = [n for n in zf.namelist() if n.lower().endswith(".png")]

            # 그룹(폴더)별로 샘플링
            groups = defaultdict(list)
            for n in pngs:
                groups[os.path.dirname(n)].append(n)
            selected = []
            for g, files in groups.items():
                files.sort()
                if sample and sample > 0 and len(files) > sample:
                    files = random.sample(files, sample)
                selected.extend(files)

            print(f"[{acup}] 그룹 {len(groups)}개, 처리 {len(selected)}장 ...", flush=True)

            for n in selected:
                key = json_key(acup, n)
                if key not in info:
                    continue
                hand_pos = info[key][0].get("hand_pos", "?")
                coord = info[key][1]["acup_coord"]

                data = zf.read(n)
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

                slot = agg[(acup, hand_pos)]
                slot[0] += 1
                row = [acup, hand_pos, key, 0, w, h, coord[0], coord[1], "", ""]
                lm_flat = [""] * 63

                if res.hand_landmarks:
                    slot[1] += 1
                    lm = res.hand_landmarks[0]
                    pts = np.array([[p.x * w, p.y * h] for p in lm])
                    uv = affine_uv(coord, pts)
                    row[3] = 1
                    if uv is not None:
                        row[8], row[9] = f"{uv[0]:.5f}", f"{uv[1]:.5f}"
                        slot[2].append(uv[0]); slot[3].append(uv[1])
                    lm_flat = []
                    for p in lm:
                        lm_flat += [f"{p.x:.5f}", f"{p.y:.5f}", f"{p.z:.5f}"]
                lm_writer.writerow(row + lm_flat)
            lm_file.flush()

    detector.close()
    lm_file.close()

    # 리포트
    rep_path = os.path.join(OUT_DIR, "report.csv")
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["acup", "hand_pos", "n", "detected", "det_rate",
                     "u_mean", "u_std", "v_mean", "v_std"])
        print("\n===== 혈자리별 리포트 =====")
        print(f"{'acup':10} {'hand_pos':14} {'n':>5} {'det%':>6} {'u_std':>7} {'v_std':>7}")
        for (acup, pos), (n, det, us, vs) in sorted(agg.items()):
            rate = 100.0 * det / max(n, 1)
            um = np.mean(us) if us else float("nan")
            ustd = np.std(us) if us else float("nan")
            vm = np.mean(vs) if vs else float("nan")
            vstd = np.std(vs) if vs else float("nan")
            wr.writerow([acup, pos, n, det, f"{rate:.1f}",
                         f"{um:.4f}", f"{ustd:.4f}", f"{vm:.4f}", f"{vstd:.4f}"])
            print(f"{acup:10} {pos:14} {n:>5} {rate:>5.1f}% {ustd:>7.3f} {vstd:>7.3f}")

    print(f"\n학습 테이블: {lm_path}")
    print(f"리포트    : {rep_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=250,
                    help="그룹(폴더)당 샘플 수. 0이면 전수")
    ap.add_argument("--acups", nargs="*", default=ALL_ACUPS)
    args = ap.parse_args()
    process(args.acups, args.sample, ZIP_DIR)
