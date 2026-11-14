"""표준화 + 로지스틱 회귀 분류기.

무게중심 분류에 쓴다. 평가(docs/evaluation.md)에서 LSTM보다 정확했고, 구조가 단순해서
"어떤 관절 좌표에 얼마의 가중치를 주는지"를 그대로 들여다볼 수 있다.

학습은 scikit-learn으로 하고, 학습된 숫자(평균, 표준편차, 가중치, 절편)만 .npz로 저장한다.
추론은 그 숫자로 직접 계산하므로 모델 객체를 pickle할 필요가 없다.

    z      = (x - mean) / scale          # 표준화: 값마다 눈금이 달라도 같은 기준으로 맞춘다
    점수   = z @ weights.T + bias         # 클래스마다 가중합
    확률   = softmax(점수)                # 합이 1이 되도록 바꾼다
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LinearClassifier:
    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        labels: list[str],
    ) -> None:
        self.mean = mean  # (특징 수,)
        self.scale = scale  # (특징 수,)
        self.weights = weights  # (클래스 수, 특징 수)
        self.bias = bias  # (클래스 수,)
        self.labels = list(labels)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, labels) -> LinearClassifier:
        """x: (N, 특징 수), y: (N,) 클래스 번호. 클래스가 3개 이상이어야 한다."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x)
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(x), y)
        return cls(scaler.mean_, scaler.scale_, model.coef_, model.intercept_, list(labels))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """클래스별 확률 (N, 클래스 수)"""
        scores = ((x - self.mean) / self.scale) @ self.weights.T + self.bias
        scores = scores - scores.max(axis=1, keepdims=True)  # 큰 값의 exp가 넘치지 않게 한다
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            bias=self.bias,
            labels=np.array(self.labels),
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearClassifier:
        with np.load(path) as data:
            return cls(
                data["mean"], data["scale"], data["weights"], data["bias"], [str(v) for v in data["labels"]]
            )
