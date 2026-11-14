"""영상 파일 이름에서 라벨을 읽고, 저장된 키포인트로 학습 데이터를 만든다."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kick_analysis import config, features, impact
from kick_analysis.extraction import VideoKeypoints

# 무게중심 영상 이름 예:
#   kick_video(back_1).mp4               → 라벨 back, 번호 1
#   kick_video_short_pass(middle_22).MOV → 라벨 middle, 번호 22, 킥 종류 short_pass
_WEIGHT_NAME = re.compile(
    r"kick_video(?:_(?P<kick>long|shoot|short_pass))?\((?P<label>back|front|mid|middle)_(?P<num>\d+)\)",
    re.IGNORECASE,
)

MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = [
    "file",
    "label",
    "kick_type",
    "clip_id",
    "source",
    "frames",
    "fps",
    "width",
    "height",
    "pose_rate",
    "ball_rate",
    "impact_frame",
]


@dataclass(frozen=True)
class ClipInfo:
    label: str  # back / front / middle (디딤발 영상이면 aligned 등)
    kick_type: str  # long / shoot / short_pass, 모르면 빈 문자열
    clip_id: int  # 파일 이름의 번호, 없으면 -1


def parse_video_name(path: str | Path) -> ClipInfo:
    """영상 경로에서 라벨을 알아낸다.

    파일 이름이 무게중심 영상 규칙에 맞으면 이름에서 읽고,
    아니면 영상이 들어 있는 폴더 이름을 라벨로 쓴다(예: data/foot_videos/aligned/1.mp4).
    """
    path = Path(path)
    match = _WEIGHT_NAME.search(path.name)
    if match:
        label = match["label"].lower()
        if label == "mid":
            label = "middle"
        kick_type = (match["kick"] or "").lower()
        return ClipInfo(label=label, kick_type=kick_type, clip_id=int(match["num"]))
    return ClipInfo(label=path.parent.name, kick_type="", clip_id=-1)


def write_manifest(keypoint_dir: str | Path, rows: list[dict]) -> Path:
    """키포인트 폴더에 영상 목록(manifest.csv)을 쓴다."""
    path = Path(keypoint_dir) / MANIFEST_NAME
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["source"], r["label"], r["clip_id"])))
    return path


def read_manifest(keypoint_dir: str | Path) -> list[dict]:
    with open(Path(keypoint_dir) / MANIFEST_NAME, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_linear_dataset(
    task: str, keypoint_dir: str | Path, source: str | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """지금 쓰는 분류기(로지스틱 회귀)의 학습 데이터 (X, y, 파일 이름).

    task가 "weight"면 X는 (N, 66), "foot"이면 (N, 6). y는 config의 라벨 순서 기준 클래스 번호.
    임팩트를 못 찾았거나 접촉 시점에 관절이 안 잡힌 영상은 건너뛴다.
    """
    labels, build = {
        "weight": (config.IMPACT_LABELS, features.build_weight_features),
        "foot": (config.FOOT_LABELS, features.build_foot_features),
    }[task]

    keypoint_dir = Path(keypoint_dir)
    xs, ys, names = [], [], []
    for row in read_manifest(keypoint_dir):
        if source and row["source"] != source:
            continue
        if row["label"] not in labels:
            continue
        keypoints = VideoKeypoints.load(keypoint_dir / row["file"])
        found = impact.find_impact(keypoints.pose, keypoints.ball)
        if found is None:
            continue
        x = build(keypoints.pose, found, keypoints.width / keypoints.height)
        if x is None:
            continue
        xs.append(x)
        ys.append(labels.index(row["label"]))
        names.append(row["file"])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64), names


# ---- 아래 두 함수는 논문 방식(화면 좌표 + LSTM)의 입력을 만든다. 비교 평가용으로 남겨 둔다 ----


def load_impact_dataset(
    keypoint_dir: str | Path, source: str | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """무게중심 모델 학습 데이터 (X, y, 파일 이름)를 만든다.

    X: (N, 6, 99), y: (N,) config.IMPACT_LABELS 기준 클래스 번호.
    source를 주면 manifest의 source 값이 같은 영상만 쓴다.
    임팩트를 못 찾았거나 앞뒤 프레임이 모자란 영상은 건너뛴다.
    """
    keypoint_dir = Path(keypoint_dir)
    xs, ys, names = [], [], []
    for row in read_manifest(keypoint_dir):
        if source and row["source"] != source:
            continue
        if row["label"] not in config.IMPACT_LABELS:
            continue
        keypoints = VideoKeypoints.load(keypoint_dir / row["file"])
        impact_frame = impact.find_impact_frame(keypoints.pose, keypoints.ball)
        if impact_frame is None:
            continue
        sequence = features.build_impact_sequence(keypoints.pose, impact_frame)
        if sequence is None:
            continue
        xs.append(sequence)
        ys.append(config.IMPACT_LABELS.index(row["label"]))
        names.append(row["file"])

    if not xs:
        return (
            np.empty((0, config.IMPACT_SEQ_LEN, config.IMPACT_INPUT_SIZE), np.float32),
            np.empty(0, np.int64),
            [],
        )
    return np.stack(xs), np.array(ys, dtype=np.int64), names


def load_foot_dataset(
    keypoint_dir: str | Path, source: str | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """디딤발 모델 학습 데이터 (X, y, 파일 이름). X: (N, 11, 305)"""
    keypoint_dir = Path(keypoint_dir)
    xs, ys, names = [], [], []
    for row in read_manifest(keypoint_dir):
        if source and row["source"] != source:
            continue
        if row["label"] not in config.FOOT_LABELS:
            continue
        keypoints = VideoKeypoints.load(keypoint_dir / row["file"])
        impact_frame = impact.find_impact_frame(keypoints.pose, keypoints.ball)
        if impact_frame is None:
            continue
        sequence = features.build_foot_sequence(keypoints.pose, keypoints.ball, impact_frame)
        if sequence is None:
            continue
        xs.append(sequence)
        ys.append(config.FOOT_LABELS.index(row["label"]))
        names.append(row["file"])

    if not xs:
        return (
            np.empty((0, config.FOOT_SEQ_LEN, config.FOOT_INPUT_SIZE), np.float32),
            np.empty(0, np.int64),
            [],
        )
    return np.stack(xs), np.array(ys, dtype=np.int64), names
