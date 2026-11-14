"""추출 결과가 영상 프레임과 1:1로 맞는지 확인한다.

실제 MediaPipe와 YOLO 대신 가짜 검출 함수를 끼워 넣어, 프레임 정렬 규칙만 빠르게 검사한다.
"""

import cv2
import numpy as np
import pytest

from kick_analysis import config
from kick_analysis.extraction import KeypointExtractor


class _FakePoseModule:
    """mp.solutions.pose 자리에 넣는 대역. Pose(...)를 with 문으로 쓸 수 있게만 한다."""

    class Pose:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False


@pytest.fixture
def tiny_video(tmp_path):
    """프레임 번호가 밝기에 적힌 10프레임짜리 영상. i번째 프레임의 밝기는 i * 20."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def _frame_number(frame) -> int:
    return int(round(frame.mean() / 20))


def _make_extractor(pose_missing: set[int], ball_missing: set[int]) -> KeypointExtractor:
    extractor = KeypointExtractor.__new__(KeypointExtractor)  # 실제 모델을 불러오지 않는다
    extractor._mp_pose = _FakePoseModule
    extractor._model_complexity = 0
    extractor._fast_complexity = None  # 기본은 모델 하나로 전체를 뽑는다
    extractor._ball_stop_after = None  # 기본은 모든 프레임에서 공을 찾는다

    def detect_pose(pose_model, frame):
        n = _frame_number(frame)
        pose = np.full((config.NUM_POSE_LANDMARKS, 4), np.nan, dtype=np.float32)
        world = np.full((config.NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
        if n not in pose_missing:
            pose[:] = n  # 값에 프레임 번호를 적어 둔다
            world[:] = n
        return pose, world

    def detect_ball(frame):
        n = _frame_number(frame)
        ball = np.full(5, np.nan, dtype=np.float32)
        if n not in ball_missing:
            ball[:] = n
        return ball

    extractor._detect_pose = detect_pose
    extractor._detect_ball = detect_ball
    return extractor


def test_rows_stay_aligned_with_frames_when_detection_fails(tiny_video):
    extractor = _make_extractor(pose_missing={2, 3}, ball_missing={7})
    keypoints = extractor.extract(tiny_video)

    # 못 찾은 프레임이 있어도 행 수는 영상 프레임 수와 같다
    assert keypoints.num_frames == 10
    assert keypoints.pose_detected.tolist() == [i not in {2, 3} for i in range(10)]
    assert keypoints.ball_detected.tolist() == [i != 7 for i in range(10)]

    # i번째 행에는 i번째 프레임의 값이 들어 있다(관절과 공이 같은 프레임을 가리킨다)
    for i in range(10):
        if i not in {2, 3}:
            assert keypoints.pose[i, 0, 0] == i
        if i != 7:
            assert keypoints.ball[i, 0] == i


def test_stops_ball_detection_once_rest_position_is_known(tiny_video):
    """공이 같은 자리에서 3번 잡히면 그 뒤 프레임에서는 공 검출을 부르지 않는다."""
    extractor = _make_extractor(pose_missing=set(), ball_missing={1})
    extractor._ball_stop_after = 3
    calls = []
    still_ball = np.array([0.5, 0.8, 0.05, 0.05, 0.9], dtype=np.float32)

    def detect_ball(frame):
        n = _frame_number(frame)
        calls.append(n)
        return np.full(5, np.nan, dtype=np.float32) if n == 1 else still_ball

    extractor._detect_ball = detect_ball
    keypoints = extractor.extract(tiny_video)

    # 0번 잡힘, 1번 놓침, 2번과 3번 잡힘 → 3번째 검출이 나온 3번 프레임까지만 공을 찾는다
    assert calls == [0, 1, 2, 3]
    assert keypoints.ball_detected.tolist() == [True, False, True, True] + [False] * 6
    assert keypoints.num_frames == 10  # 행 수는 그대로다
    assert keypoints.pose_detected.all()  # 관절은 끝까지 뽑는다


def test_keeps_detecting_ball_while_it_moves(tiny_video):
    """공이 계속 움직이면(같은 자리에 모이지 않으면) 끝까지 찾는다."""
    extractor = _make_extractor(pose_missing=set(), ball_missing=set())
    extractor._ball_stop_after = 3
    calls = []

    def detect_ball(frame):
        n = _frame_number(frame)
        calls.append(n)
        return np.array([0.1 + 0.08 * n, 0.8, 0.05, 0.05, 0.9], dtype=np.float32)

    extractor._detect_ball = detect_ball
    extractor.extract(tiny_video)
    assert calls == list(range(10))


# ---- 2단계 추출: 빠른 모델로 전체를 훑고, 임팩트 주변만 정밀한 모델로 다시 뽑는다 ----


class _RecordingPoseModule:
    """Pose(model_complexity=...)로 만든 모델이 자기 크기를 기억하게 한다."""

    class Pose(_FakePoseModule.Pose):
        def __init__(self, **kwargs):
            self.model_complexity = kwargs["model_complexity"]


@pytest.fixture
def long_video(tmp_path):
    """30프레임짜리 영상. i번째 프레임의 밝기는 i * 8."""
    path = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def _make_two_pass_extractor(fast_impact: int, precise_impact: int) -> KeypointExtractor:
    """발끝이 impact 프레임에서 공 중심에 오는 가짜 추출기. 모델 크기마다 impact가 다를 수 있다.

    어느 모델이 만든 행인지 알 수 있도록 visibility 칸에 모델 크기(0 또는 2)를 적는다.
    """
    extractor = KeypointExtractor.__new__(KeypointExtractor)
    extractor._mp_pose = _RecordingPoseModule
    extractor._model_complexity = 2
    extractor._fast_complexity = 0
    extractor._ball_stop_after = None

    def detect_pose(pose_model, frame):
        n = int(round(frame.mean() / 8))
        impact_frame = fast_impact if pose_model.model_complexity == 0 else precise_impact
        pose = np.full((config.NUM_POSE_LANDMARKS, 4), 0.1, dtype=np.float32)
        pose[:, 3] = pose_model.model_complexity
        pose[config.RIGHT_FOOT_INDEX, :2] = (0.5 + 0.05 * (n - impact_frame), 0.8)
        return pose, np.zeros((config.NUM_POSE_LANDMARKS, 3), dtype=np.float32)

    extractor._detect_pose = detect_pose
    extractor._detect_ball = lambda frame: np.array([0.5, 0.8, 0.05, 0.05, 0.9], dtype=np.float32)
    return extractor


def test_two_pass_refines_only_around_impact(long_video):
    extractor = _make_two_pass_extractor(fast_impact=20, precise_impact=20)
    keypoints = extractor.extract(long_video)

    # 발끝은 19번과 20번 프레임 사이에서 공에 닿는다 → 접촉 직전 프레임은 19번
    start = 19 - config.REFINE_BEFORE
    end = 19 + config.REFINE_AFTER + 1
    made_by = keypoints.pose[:, 0, 3]  # 각 행을 만든 모델의 크기
    assert keypoints.num_frames == 30
    assert (made_by[start:end] == 2).all()  # 임팩트 주변은 정밀한 모델
    assert (made_by[:start] == 0).all()  # 나머지는 빠른 모델 그대로
    assert (made_by[end:] == 0).all()


def test_two_pass_redoes_everything_when_fast_model_misses_impact(long_video):
    """빠른 모델이 임팩트를 15프레임 일찍 짚으면, 다시 뽑은 구간에 진짜 임팩트가 없다.

    그 경우 전체를 정밀한 모델로 다시 뽑아야 분류에 쓰는 프레임이 정밀한 좌표가 된다.
    """
    extractor = _make_two_pass_extractor(fast_impact=5, precise_impact=20)
    keypoints = extractor.extract(long_video)
    assert (keypoints.pose[:, 0, 3] == 2).all()


def test_two_pass_progress_reaches_total(long_video):
    extractor = _make_two_pass_extractor(fast_impact=20, precise_impact=20)
    calls = []
    extractor.extract(long_video, on_progress=lambda done, total: calls.append((done, total)))
    assert calls[-1][0] == calls[-1][1]  # 마지막에는 100%
    assert all(done <= total for done, total in calls)


def test_reports_video_properties_and_progress(tiny_video):
    extractor = _make_extractor(pose_missing=set(), ball_missing=set())
    calls = []
    keypoints = extractor.extract(tiny_video, on_progress=lambda done, total: calls.append((done, total)))

    assert (keypoints.width, keypoints.height) == (64, 48)
    assert keypoints.fps == pytest.approx(24, abs=0.5)
    assert calls[-1][0] == 10
    assert len(calls) == 10


def test_unreadable_video_raises(tmp_path):
    extractor = _make_extractor(pose_missing=set(), ball_missing=set())
    with pytest.raises(OSError):
        extractor.extract(tmp_path / "없는파일.mp4")
