"""분류 모델 정의와 저장, 불러오기, 양자화."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class KickLSTM(nn.Module):
    """LSTM 한 층 + 선형 층. 무게중심 모델과 디딤발 모델이 입력 크기만 다르게 같이 쓴다.

    시퀀스를 끝까지 읽은 뒤의 마지막 hidden 상태 하나로 클래스를 정한다.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def quantize(model: nn.Module) -> nn.Module:
    """동적 양자화: LSTM과 Linear의 가중치를 float32에서 int8로 바꾼다.

    가중치는 미리 정수로 바꿔 두고, 활성값은 추론할 때마다 범위를 재서 변환한다.
    보정용 데이터나 재학습이 필요 없어서 학습이 끝난 모델에 바로 적용할 수 있다.
    """
    model.eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)


def save_checkpoint(model: KickLSTM, labels: tuple[str, ...], path: str | Path) -> None:
    """가중치와 함께 모델 크기, 라벨 순서를 저장한다.

    모델 객체를 통째로 pickle하지 않고 가중치(state_dict)만 저장한다.
    통째로 저장하면 클래스 정의 위치나 PyTorch 버전이 바뀔 때 불러오지 못한다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": model.lstm.input_size,
            "hidden_size": model.lstm.hidden_size,
            "labels": list(labels),
        },
        path,
    )


def load_checkpoint(path: str | Path, quantized: bool = False) -> tuple[nn.Module, list[str]]:
    """저장된 모델과 라벨 목록을 불러온다. quantized=True면 양자화한 모델을 돌려준다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = KickLSTM(checkpoint["input_size"], checkpoint["hidden_size"], len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if quantized:
        model = quantize(model)
    return model, labels
