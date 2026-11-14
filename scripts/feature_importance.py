"""무게중심 분류기가 어느 관절을 많이 보는지 가중치로 살펴본다.

사용 예 (프로젝트 루트에서, 학습을 먼저 해야 한다):
    python -m scripts.feature_importance

로지스틱 회귀는 입력 값마다 가중치를 곱해 더하는 모델이라, 가중치의 크기가 곧 그 값의 영향력이다.
(입력을 표준화한 뒤 학습하므로 값마다 눈금이 달라도 가중치끼리 비교할 수 있다.)
관절마다 x, y 두 값이 있으므로 클래스 3개 x 좌표 2개의 가중치 절댓값을 평균해 관절별 값으로 만든다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kick_analysis import config
from kick_analysis.linear import LinearClassifier

# BlazePose 관절 이름 (33개)
JOINT_NAMES = [
    "Nose", "L Eye Inner", "L Eye", "L Eye Outer", "R Eye Inner", "R Eye", "R Eye Outer",
    "L Ear", "R Ear", "Mouth L", "Mouth R", "L Shoulder", "R Shoulder",
    "L Elbow", "R Elbow", "L Wrist", "R Wrist",
    "L Pinky", "R Pinky", "L Index", "R Index", "L Thumb", "R Thumb",
    "L Hip", "R Hip", "L Knee", "R Knee", "L Ankle", "R Ankle",
    "L Heel", "R Heel", "L FootIdx", "R FootIdx",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="관절별 중요도(가중치 크기)")
    parser.add_argument("--model", type=Path, default=Path("models/weights/weight_linear.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/weight_joint_importance.png"))
    args = parser.parse_args()

    model = LinearClassifier.load(args.model)
    # weights: (클래스 3, 입력 66) → (클래스 3, 관절 33, 좌표 2) → 관절별 평균
    weights = np.abs(model.weights).reshape(len(model.labels), config.NUM_POSE_LANDMARKS, 2)
    importance = weights.mean(axis=(0, 2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 6))
    plt.bar(JOINT_NAMES, importance)
    plt.title("Joint importance (mean |weight|)")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print(f"저장: {args.out}")
    for i in importance.argsort()[::-1][:8]:
        print(f"  {JOINT_NAMES[i]}: {importance[i]:.3f}")


if __name__ == "__main__":
    main()
