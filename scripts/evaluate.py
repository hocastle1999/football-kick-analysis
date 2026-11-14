"""여러 방법을 같은 기준으로 비교한다.

사용 예 (프로젝트 루트에서):
    python -m scripts.evaluate --task impact
    python -m scripts.evaluate --task foot
    python -m scripts.evaluate --task foot --exclude-label aligned

평가 방식
- 반복 교차 검증: 데이터를 5조각으로 나눠 4조각으로 학습하고 1조각으로 시험하기를 돌아가며 한다.
  나누는 방법을 바꿔 가며 여러 번 반복해 평균과 흔들림(표준편차)을 본다.
  한 번만 나누면 테스트가 20여 개뿐이라 어떻게 나뉘었는지에 따라 결과가 크게 달라진다.
- 사람 단위: 한 사람의 영상을 전부 빼고 학습한 뒤 그 사람으로 시험한다.
  같은 사람이 학습과 시험에 함께 있으면 자세가 아니라 사람(체형, 옷)을 외워도 점수가 나오기 때문이다.
- 추가 영상: 논문 영상 63개로 학습하고, 학습에 쓰지 않은 60개로 시험한다(무게중심만).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kick_analysis import config, dataset, features, impact
from kick_analysis.extraction import VideoKeypoints
from scripts.train_lstm import predict, train_model


class LstmClassifier:
    """scripts.train_lstm의 LSTM 학습을 fit / predict 형태로 감싼 것."""

    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def fit(self, x, y):
        self.model, _ = train_model(x, y, int(y.max()) + 1, self.epochs, config.SEED)
        return self

    def predict(self, x):
        return predict(self.model, x)


def logistic():
    # 표준화(평균 0, 표준편차 1) 후 로지스틱 회귀. 값 몇 개에 가중치를 곱해 더하는 가장 단순한 분류기다
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def person_of(task: str, row: dict) -> str:
    """영상을 찍은 사람. 임팩트 장면을 눈으로 확인해 정한 대응이다(docs/evaluation.md)."""
    number = int(re.search(r"(\d+)\)", row["file"]).group(1))
    if task == "impact" and row["source"] == "extra60":  # 추가 영상: 22~31, 32~41
        if number >= 32:
            return "E"
        return "C" if row["label"] == "middle" else "D"
    if task == "impact":
        return "ABC"[(number - 1) // 7]  # 논문 영상: 1~7, 8~14, 15~21
    if row["label"] == "aligned":  # aligned는 무게중심 촬영 때의 영상이다
        return "B" if number <= 3 else "C"
    return "D" if number <= 5 else "E"


def load_clips(task: str, keypoint_dir: Path, ball_every: int = 1) -> list[dict]:
    labels = config.IMPACT_LABELS if task == "impact" else config.FOOT_LABELS
    clips = []
    for row in dataset.read_manifest(keypoint_dir):
        if row["label"] not in labels:
            continue
        keypoints = VideoKeypoints.load(keypoint_dir / row["file"])
        if ball_every > 1:
            # 공을 N프레임마다 한 번만 검출했을 때를 흉내 낸다: 나머지 프레임의 검출 결과를 지운다
            skipped = np.arange(keypoints.num_frames) % ball_every != 0
            keypoints.ball[skipped] = np.nan
        clips.append(
            {
                "keypoints": keypoints,
                "label": labels.index(row["label"]),
                "source": row["source"],
                "person": person_of(task, row),
                "aspect": keypoints.width / keypoints.height,
                "old_impact": impact.find_impact_frame(keypoints.pose, keypoints.ball),
                "new_impact": impact.find_impact(keypoints.pose, keypoints.ball),
            }
        )
    return clips


# ---- 입력을 만드는 방법들: 영상 하나를 받아 모델 입력을 돌려주고, 만들 수 없으면 None ----
def raw_sequence_old(task):
    def build(c):
        k = c["keypoints"]
        if task == "impact":
            return features.build_impact_sequence(k.pose, c["old_impact"])
        return features.build_foot_sequence(k.pose, k.ball, c["old_impact"])

    return build


def raw_sequence_new(task):
    def build(c):
        k = c["keypoints"]
        if task == "impact":
            return features.build_impact_sequence(k.pose, c["new_impact"].frame)
        return features.build_foot_sequence(k.pose, k.ball, c["new_impact"].frame)

    return build


def normalized_sequence(task):
    if task == "impact":
        before, length = config.IMPACT_SEQ_BEFORE, config.IMPACT_SEQ_LEN
    else:
        before, length = config.FOOT_SEQ_BEFORE, config.FOOT_SEQ_LEN
    return lambda c: features.build_normalized_sequence(
        c["keypoints"].pose, c["new_impact"], c["aspect"], before, length
    )


def normalized_pose(task):
    def build(c):
        pose = features.build_normalized_pose(c["keypoints"].pose, c["new_impact"], c["aspect"])
        return None if pose is None else pose.reshape(-1)

    return build


def geometry(task):
    pick = features.weight_geometry if task == "impact" else features.foot_geometry

    def build(c):
        pose = features.build_normalized_pose(c["keypoints"].pose, c["new_impact"], c["aspect"])
        return None if pose is None else pick(pose)

    return build


METHODS = [
    ("논문 방식: 화면 좌표 + LSTM 50에폭", raw_sequence_old, lambda: LstmClassifier(50)),
    ("화면 좌표 + LSTM 300에폭", raw_sequence_old, lambda: LstmClassifier(300)),
    ("새 임팩트 + 화면 좌표 + LSTM 300에폭", raw_sequence_new, lambda: LstmClassifier(300)),
    ("새 임팩트 + 정규화 좌표 + LSTM 300에폭", normalized_sequence, lambda: LstmClassifier(300)),
    ("새 임팩트 + 정규화 자세 1장 + 로지스틱", normalized_pose, logistic),
    ("새 임팩트 + 기하 특징 + 로지스틱", geometry, logistic),
]


def accuracy(model, x_train, y_train, x_test, y_test) -> float:
    return float((model.fit(x_train, y_train).predict(x_test) == y_test).mean())


def evaluate(clips: list[dict], build, make_model, repeats: int) -> dict:
    """한 방법을 세 가지 방식으로 평가한다. 입력을 만들 수 없는 영상은 뺀다."""
    usable = [(build(c), c) for c in clips]
    usable = [(x, c) for x, c in usable if x is not None]
    x = np.stack([x for x, _ in usable])
    y = np.array([c["label"] for _, c in usable])
    person = np.array([c["person"] for _, c in usable])
    extra = np.array([c["source"] == "extra60" for _, c in usable])
    result = {"n": len(x)}

    # 반복 교차 검증: 반복 한 번(5조각)의 평균을 구하고, 반복들 사이의 평균과 표준편차를 낸다
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
    folds = [accuracy(make_model(), x[a], y[a], x[b], y[b]) for a, b in splitter.split(x, y)]
    per_repeat = np.array(folds).reshape(repeats, 5).mean(axis=1)
    result["cv"] = (per_repeat.mean(), per_repeat.std())

    # 사람 단위: 학습 쪽에 모든 클래스가 남아 있을 때만 의미가 있다
    correct = total = 0
    result["per_person"] = {}
    for a, b in LeaveOneGroupOut().split(x, y, person):
        if len(set(y[a])) < len(set(y)):
            correct = total = 0
            break
        score = accuracy(make_model(), x[a], y[a], x[b], y[b])
        result["per_person"][person[b][0]] = score
        correct += score * len(b)
        total += len(b)
    result["person"] = correct / total if total else None

    # 논문 영상으로만 학습하고 추가 영상으로 시험(촬영 날짜와 해상도가 다른 영상에 대한 성능)
    result["extra"] = (
        accuracy(make_model(), x[~extra], y[~extra], x[extra], y[extra]) if extra.any() else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="방법별 정확도 비교")
    parser.add_argument("--task", choices=["impact", "foot"], required=True)
    parser.add_argument("--keypoints", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--exclude-label", help="이 라벨을 빼고 평가한다(예: aligned)")
    parser.add_argument("--ball-every", type=int, default=1, help="공을 N프레임마다 한 번만 검출했다고 가정")
    parser.add_argument("--current-only", action="store_true", help="지금 쓰는 방식(로지스틱 회귀)만 평가")
    args = parser.parse_args()

    default_dir = "data/keypoints/weight" if args.task == "impact" else "data/keypoints/foot"
    clips = load_clips(args.task, args.keypoints or Path(default_dir), args.ball_every)
    methods = METHODS[-2:] if args.current_only else METHODS
    if args.exclude_label:
        labels = config.FOOT_LABELS if args.task == "foot" else config.IMPACT_LABELS
        removed = labels.index(args.exclude_label)
        clips = [c for c in clips if c["label"] != removed]
        for c in clips:  # 클래스 번호를 0부터 다시 매긴다
            c["label"] -= c["label"] > removed

    print("| 방법 | 영상 수 | 반복 교차 검증 | 사람 단위 | 사람별 | 63개로 학습 → 추가 60개 |")
    print("|---|---|---|---|---|---|")
    for name, make_builder, make_model in methods:
        r = evaluate(clips, make_builder(args.task), make_model, args.repeats)
        person = "-" if r["person"] is None else f"{r['person']:.2f}"
        each = " ".join(f"{p}={s:.2f}" for p, s in sorted(r["per_person"].items())) if r["person"] else "-"
        extra = "-" if r["extra"] is None else f"{r['extra']:.2f}"
        cv = f"{r['cv'][0]:.2f} ± {r['cv'][1]:.2f}"
        print(f"| {name} | {r['n']} | {cv} | {person} | {each} | {extra} |", flush=True)


if __name__ == "__main__":
    main()
