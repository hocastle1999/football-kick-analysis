"""무게중심 모델(impact) 또는 디딤발 모델(foot)을 학습하고 평가한다.

사용 예 (프로젝트 루트에서):
    python -m scripts.train_lstm --task impact --source thesis63
    python -m scripts.train_lstm --task foot --keypoints data/keypoints/foot

학습 설정은 논문과 같다: 전체 데이터를 한 번에 넣는 방식으로 50 에폭, Adam(lr=0.001), hidden 64.
평가는 무작위 60/40 분할 한 번이다. (교차 검증과 사람 단위 분할은 다음 단계에서 추가한다.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from kick_analysis import config, dataset
from kick_analysis.models import KickLSTM, quantize, save_checkpoint

TASKS = {
    "impact": (dataset.load_impact_dataset, config.IMPACT_LABELS, "data/keypoints/weight"),
    "foot": (dataset.load_foot_dataset, config.FOOT_LABELS, "data/keypoints/foot"),
}


def train_model(x_train: np.ndarray, y_train: np.ndarray, num_classes: int, epochs: int, seed: int):
    """모델을 학습해 (모델, 에폭별 손실)을 돌려준다."""
    torch.manual_seed(seed)  # 가중치 초기값을 고정해 실행할 때마다 같은 결과가 나오게 한다
    model = KickLSTM(x_train.shape[2], config.HIDDEN_SIZE, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return model, losses


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).argmax(dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="킥 분석 LSTM 학습")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--keypoints", type=Path, help="키포인트 폴더(기본값은 task에 따라 정해진다)")
    parser.add_argument("--source", help="manifest의 source가 이 값인 영상만 사용")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=Path("models/weights"))
    args = parser.parse_args()

    load_dataset, labels, default_dir = TASKS[args.task]
    x, y, names = load_dataset(args.keypoints or Path(default_dir), source=args.source)
    if len(x) == 0:
        raise SystemExit("학습할 데이터가 없습니다. 먼저 scripts.extract_keypoints를 실행하세요.")
    counts = {label: int((y == i).sum()) for i, label in enumerate(labels)}
    print(f"데이터 {len(x)}개, 클래스별 {counts}")

    # stratify: 학습과 테스트의 클래스 비율을 같게 나눈다
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    model, losses = train_model(x_train, y_train, len(labels), args.epochs, args.seed)
    print(f"학습 {len(x_train)}개, 테스트 {len(x_test)}개, 마지막 손실 {losses[-1]:.4f}")

    # 실제 추론에 쓰는 것은 양자화 모델이므로 양자화 전후를 같은 테스트셋으로 모두 평가한다
    for title, evaluated in (("FP32(양자화 전)", model), ("INT8(동적 양자화 후)", quantize(model))):
        print(f"\n[{title}]")
        print(
            classification_report(
                y_test,
                predict(evaluated, x_test),
                labels=list(range(len(labels))),
                target_names=labels,
                zero_division=0,
            )
        )

    checkpoint_path = args.out / f"{args.task}_lstm.pt"
    save_checkpoint(model, labels, checkpoint_path)
    print(f"모델 저장: {checkpoint_path}")

    plot_path = Path("outputs") / f"{args.task}_lstm_loss.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.title(f"{args.task} LSTM loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"손실 그래프 저장: {plot_path}")


if __name__ == "__main__":
    main()
