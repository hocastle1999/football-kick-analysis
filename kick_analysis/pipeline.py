"""영상 하나를 분석하는 전체 흐름: 좌표 추출 → 임팩트 추정 → 분류 → 피드백.

웹 서버, GUI, 명령줄 스크립트가 이 모듈을 같이 쓴다. 화면 표시와 관련된 코드는 여기에 두지 않는다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kick_analysis import features, impact
from kick_analysis.extraction import KeypointExtractor, VideoKeypoints
from kick_analysis.linear import LinearClassifier

# 킥 종류별로 권장하는 무게중심 위치
RECOMMENDED_WEIGHT = {"long": "back", "short_pass": "middle", "shoot": "front"}
IDEAL_FOOT = "aligned"


class AnalysisError(Exception):
    """영상을 분석할 수 없을 때(공을 못 찾음 등). 메시지는 사용자에게 그대로 보여 준다."""


@dataclass
class AnalysisResult:
    impact_frame: int  # 발이 공에 닿기 직전의 마지막 프레임(화면 표시용). 공이 아직 제자리에 있다
    impact_time: float  # 프레임 단위의 접촉 시점. 23.4면 23번과 24번 프레임 사이
    weight_label: str | None  # back / front / middle, 판단하지 못하면 None
    weight_confidence: float | None  # 분류기가 그 라벨에 준 확률(0~1)
    foot_label: str | None  # aligned 등, 모델이 없거나 판단하지 못하면 None
    foot_confidence: float | None
    keypoints: VideoKeypoints
    timings: dict[str, float] = field(default_factory=dict)  # 단계별 소요 시간(초)


class KickAnalyzer:
    """모델을 한 번 불러 두고 여러 영상을 분석한다."""

    def __init__(
        self, weight_model_path: str | Path | None, foot_model_path: str | Path | None = None
    ) -> None:
        self.extractor = KeypointExtractor()
        self.weight_model = self._load(weight_model_path)
        self.foot_model = self._load(foot_model_path)
        if self.weight_model is None and self.foot_model is None:
            raise FileNotFoundError("불러올 모델이 없습니다. 먼저 scripts.train으로 학습하세요.")

    @staticmethod
    def _load(path) -> LinearClassifier | None:
        if path is None or not Path(path).exists():
            return None
        return LinearClassifier.load(path)

    def analyze(
        self,
        video_path: str | Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> AnalysisResult:
        started = time.perf_counter()
        keypoints = self.extractor.extract(video_path, on_progress)
        extract_seconds = time.perf_counter() - started

        classify_started = time.perf_counter()
        found = impact.find_impact(keypoints.pose, keypoints.ball)
        if found is None:
            raise AnalysisError("영상에서 공이나 차는 발을 찾지 못해 임팩트를 정할 수 없습니다.")

        aspect = keypoints.width / keypoints.height
        weight_label, weight_confidence = self._classify(
            self.weight_model, features.build_weight_features(keypoints.pose, found, aspect)
        )
        foot_label, foot_confidence = self._classify(
            self.foot_model, features.build_foot_features(keypoints.pose, found, aspect)
        )
        finished = time.perf_counter()

        return AnalysisResult(
            # 접촉 직후 프레임을 보여 주면 공이 이미 떠나 있어서, 표시한 공의 자리와 어긋난다
            impact_frame=found.frame,
            impact_time=found.time,
            weight_label=weight_label,
            weight_confidence=weight_confidence,
            foot_label=foot_label,
            foot_confidence=foot_confidence,
            keypoints=keypoints,
            timings={
                "extract": extract_seconds,
                "classify": finished - classify_started,  # 임팩트 추정 + 분류
                "total": finished - started,
            },
        )

    @staticmethod
    def _classify(model: LinearClassifier | None, x: np.ndarray | None) -> tuple[str | None, float | None]:
        """입력 하나를 분류해 (라벨, 확률)을 돌려준다. 모델이나 입력이 없으면 (None, None)."""
        if model is None or x is None:  # x가 None: 접촉 시점에 관절이 안 잡힌 경우
            return None, None
        proba = model.predict_proba(x[None, :])[0]
        best = int(proba.argmax())
        return model.labels[best], float(proba[best])


def make_feedback(kick_type: str, weight_label: str | None, foot_label: str | None) -> list[str]:
    """분류 결과를 킥 종류별 권장 자세와 비교해 피드백 문장을 만든다."""
    messages = []

    recommended = RECOMMENDED_WEIGHT[kick_type]
    if weight_label is None:
        messages.append("무게중심: 판단하지 못했습니다")
    elif weight_label == recommended:
        messages.append(f"무게중심: 이상적입니다 (현재: {weight_label})")
    else:
        messages.append(f"무게중심: {recommended} 권장 (현재: {weight_label})")

    if foot_label is None:
        messages.append("디딤발: 판단하지 못했습니다")
    elif foot_label == IDEAL_FOOT:
        messages.append(f"디딤발: 이상적입니다 (현재: {foot_label})")
    else:
        messages.append(f"디딤발: {IDEAL_FOOT} 권장 (현재: {foot_label})")

    return messages
