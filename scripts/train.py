"""분석에 쓰는 분류기(로지스틱 회귀)를 학습해 저장한다.

사용 예 (프로젝트 루트에서):
    python -m scripts.train --task weight
    python -m scripts.train --task foot

가진 영상을 전부 써서 학습한다. 성능은 여기서 재지 않는다.
학습에 쓴 데이터로 잰 정확도는 실제 성능보다 높게 나오기 때문이다.
성능 비교는 scripts.evaluate가 교차 검증과 사람 단위 평가로 한다(docs/evaluation.md).

논문 방식(LSTM)은 scripts.train_lstm에 비교용으로 남아 있다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kick_analysis import config, dataset
from kick_analysis.linear import LinearClassifier

TASKS = {
    "weight": (config.IMPACT_LABELS, "data/keypoints/weight"),
    "foot": (config.FOOT_LABELS, "data/keypoints/foot"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="킥 분석 분류기 학습")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--keypoints", type=Path, help="키포인트 폴더(기본값은 task에 따라 정해진다)")
    parser.add_argument("--source", help="manifest의 source가 이 값인 영상만 사용")
    parser.add_argument("--out", type=Path, default=Path("models/weights"))
    args = parser.parse_args()

    labels, default_dir = TASKS[args.task]
    x, y, _ = dataset.load_linear_dataset(args.task, args.keypoints or Path(default_dir), args.source)
    if len(x) == 0:
        raise SystemExit("학습할 데이터가 없습니다. 먼저 scripts.extract_keypoints를 실행하세요.")
    counts = {label: int((y == i).sum()) for i, label in enumerate(labels)}
    print(f"데이터 {len(x)}개, 입력 값 {x.shape[1]}개, 클래스별 {counts}")

    model = LinearClassifier.fit(x, y, labels)
    path = args.out / f"{args.task}_linear.npz"
    model.save(path)
    print(f"학습 데이터에 대한 정확도: {(model.predict(x) == y).mean():.2f} (실제 성능이 아니다)")
    print(f"모델 저장: {path}")


if __name__ == "__main__":
    main()
