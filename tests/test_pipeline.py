import numpy as np
import pytest

from kick_analysis import config
from kick_analysis.extraction import VideoKeypoints
from kick_analysis.linear import LinearClassifier
from kick_analysis.pipeline import AnalysisError, KickAnalyzer, make_feedback


@pytest.mark.parametrize(
    ("kick_type", "ideal"),
    [("long", "back"), ("short_pass", "middle"), ("shoot", "front")],
)
def test_weight_feedback_matches_kick_type(kick_type, ideal):
    good = make_feedback(kick_type, ideal, "aligned")
    assert good == [f"무게중심: 이상적입니다 (현재: {ideal})", "디딤발: 이상적입니다 (현재: aligned)"]

    other = "front" if ideal != "front" else "back"
    bad = make_feedback(kick_type, other, "aligned")
    assert bad[0] == f"무게중심: {ideal} 권장 (현재: {other})"


def test_foot_feedback():
    assert make_feedback("long", "back", "too_far")[1] == "디딤발: aligned 권장 (현재: too_far)"


def test_feedback_when_model_could_not_decide():
    assert make_feedback("shoot", None, None) == [
        "무게중심: 판단하지 못했습니다",
        "디딤발: 판단하지 못했습니다",
    ]


# ---- 분석 흐름: 실제 포즈 모델 대신 정해진 좌표를 돌려주는 가짜 추출기를 쓴다 ----


class FakeExtractor:
    def __init__(self, pose, ball):
        world = np.zeros((len(pose), config.NUM_POSE_LANDMARKS, 3), dtype=np.float32)
        self.keypoints = VideoKeypoints(pose, world, ball, fps=24.0, width=1280, height=720)

    def extract(self, video_path, on_progress=None):
        return self.keypoints


def make_analyzer(pose, ball, weight_model=None, foot_model=None) -> KickAnalyzer:
    analyzer = KickAnalyzer.__new__(KickAnalyzer)  # 실제 모델 파일을 불러오지 않는다
    analyzer.extractor = FakeExtractor(pose, ball)
    analyzer.weight_model = weight_model
    analyzer.foot_model = foot_model
    return analyzer


def lean_classifier() -> LinearClassifier:
    """어깨가 골반보다 뒤면 back, 앞이면 front로 가르는 분류기를 가짜 데이터로 학습한다."""
    from kick_analysis import features
    from kick_analysis.impact import find_impact

    xs, ys = [], []
    for label, lean in ((0, -0.08), (1, 0.08), (2, 0.0)):  # back, front, middle
        for jitter in np.linspace(-0.01, 0.01, 5):
            pose, ball = _clip(lean + jitter)
            xs.append(features.build_weight_features(pose, find_impact(pose, ball), 16 / 9))
            ys.append(label)
    return LinearClassifier.fit(np.array(xs), np.array(ys), config.IMPACT_LABELS)


def _clip(lean: float):
    pose = np.zeros((20, config.NUM_POSE_LANDMARKS, 4), dtype=np.float32)
    pose[:, :, :2] = 0.1
    for t in range(20):
        pose[t, config.RIGHT_FOOT_INDEX, :2] = (0.5 + 0.05 * (t - 10), 0.8)
    pose[:, [23, 24], :2] = (0.42, 0.60)
    pose[:, [11, 12], :2] = (0.42 + lean, 0.40)
    ball = np.tile(np.array([0.5, 0.8, 0.05, 0.05, 0.9], dtype=np.float32), (20, 1))
    return pose, ball


def test_analyze_classifies_posture_at_impact():
    pose, ball = _clip(lean=-0.08)  # 뒤로 기운 자세
    result = make_analyzer(pose, ball, weight_model=lean_classifier()).analyze("clip.mp4")

    assert result.impact_time == pytest.approx(9.5, abs=1e-3)
    assert result.impact_frame in (9, 10)
    assert result.weight_label == "back"
    assert 0.5 < result.weight_confidence <= 1.0
    assert (result.foot_label, result.foot_confidence) == (None, None)  # 디딤발 모델이 없다
    assert set(result.timings) == {"extract", "classify", "total"}


def test_analyze_fails_clearly_without_ball():
    pose, ball = _clip(lean=0.0)
    ball[:] = np.nan
    with pytest.raises(AnalysisError):
        make_analyzer(pose, ball, weight_model=lean_classifier()).analyze("clip.mp4")


def test_analyze_skips_classification_when_pose_missing_at_impact():
    pose, ball = _clip(lean=0.0)
    pose[9:11, :23] = np.nan  # 접촉 순간 상체가 안 잡힌 경우
    result = make_analyzer(pose, ball, weight_model=lean_classifier()).analyze("clip.mp4")
    assert result.weight_label is None
    assert result.weight_confidence is None
