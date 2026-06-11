"""
혈자리 추론 코어 — 손 사진 한 장 -> 21 랜드마크(MediaPipe) -> M2 -> 11혈 좌표.

앱/서버 어디서든 재사용하는 단일 진입점:
    predictor = AcupointPredictor()
    result = predictor.predict(img_bgr)        # dict (JSON 직렬화 가능)
    overlay = predictor.overlay(img_bgr, result)   # 시각화용 BGR ndarray
"""
import os
from dataclasses import dataclass
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASK = os.path.join(HERE, "models", "hand_landmarker.task")
DEFAULT_M2 = os.path.join(HERE, "models", "m2_model.joblib")

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17

# 혈자리 메타 (로마자키 -> 한글명, 경혈코드)
ACUP_META = {
    "ekmoon":    ("액문", "TE2"),
    "gwanchung": ("관충", "TE1"),
    "hapgok":    ("합곡", "LI4"),
    "hugye":     ("후계", "SI3"),
    "jungjer":   ("중저", "TE3"),
    "nogung":    ("노궁", "PC8"),
    "sangyang":  ("상양", "LI1"),
    "sochung":   ("소충", "HT9"),
    "sotack":    ("소택", "SI1"),
    "taeyeon":   ("태연", "LU9"),
    "urjae":     ("어제", "LU10"),
}


@dataclass
class _Basis:
    o: np.ndarray
    s: float
    e1: np.ndarray
    e2: np.ndarray


class AcupointPredictor:
    def __init__(self, task_path: str = DEFAULT_TASK, m2_path: str = DEFAULT_M2,
                 min_conf: float = 0.3):
        bundle = joblib.load(m2_path)
        self.m2 = bundle["model"]
        self.acups = bundle["acups"]
        self._eye = np.eye(len(self.acups))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1, min_hand_detection_confidence=min_conf,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(opts)

    # ---- 내부 기하 ----
    @staticmethod
    def _basis(lm: np.ndarray) -> _Basis:
        o = lm[WRIST]
        r = lm[MIDDLE_MCP] - o
        s = max(float(np.linalg.norm(r)), 1e-6)
        e1 = r / s
        e2 = np.array([-e1[1], e1[0]])
        vi, vp = lm[INDEX_MCP] - o, lm[PINKY_MCP] - o
        if vi[0] * vp[1] - vi[1] * vp[0] < 0:   # 좌/우손 chirality 정규화
            e2 = -e2
        return _Basis(o, s, e1, e2)

    def _landmarks(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            return None, None
        h, w = img_bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
        handed = "unknown"
        if res.handedness:
            handed = res.handedness[0][0].category_name.lower()  # 'left'/'right'
        return pts, handed

    # ---- 공개 API ----
    def predict(self, img_bgr) -> dict:
        h, w = img_bgr.shape[:2]
        lm, handed = self._landmarks(img_bgr)
        if lm is None:
            return {"detected": False, "image_size": [int(w), int(h)],
                    "handedness": None, "landmarks": [], "acupoints": []}
        b = self._basis(lm)
        canon = np.array([[(p - b.o) @ b.e1 / b.s, (p - b.o) @ b.e2 / b.s] for p in lm])
        feat = canon.reshape(-1)
        X = np.stack([np.concatenate([feat, self._eye[i]]) for i in range(len(self.acups))])
        pred = self.m2.predict(X)
        acupoints = []
        for i, a in enumerate(self.acups):
            px = b.o + b.s * (pred[i, 0] * b.e1 + pred[i, 1] * b.e2)
            kr, code = ACUP_META.get(a, (a, ""))
            acupoints.append({"name": a, "name_kr": kr, "code": code,
                              "x": round(float(px[0]), 1), "y": round(float(px[1]), 1)})
        return {"detected": True, "image_size": [int(w), int(h)],
                "handedness": handed,
                "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in lm],
                "acupoints": acupoints}

    def overlay(self, img_bgr, result: dict):
        import colorsys
        vis = img_bgr.copy()
        if not result.get("detected"):
            cv2.putText(vis, "No hand detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return vis
        n = len(result["acupoints"])
        for i, ap in enumerate(result["acupoints"]):
            r, g, bl = colorsys.hsv_to_rgb(i / n, 0.9, 1.0)
            c = (int(bl * 255), int(g * 255), int(r * 255))
            x, y = int(ap["x"]), int(ap["y"])
            cv2.circle(vis, (x, y), 6, c, -1)
            cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
            cv2.putText(vis, f"{ap['name']} {ap['code']}", (x + 7, y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        return vis
