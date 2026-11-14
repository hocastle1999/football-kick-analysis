import numpy as np

from kick_analysis import config
from kick_analysis.impact import find_impact_frame


def test_finds_frame_where_foot_meets_ball(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=12)
    assert find_impact_frame(pose, ball) == 12


def test_ignores_frames_without_ball(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    ball[9:12] = np.nan  # 임팩트 주변에서 공을 놓친 경우
    # 남은 후보 중 발과 공이 가장 가까운 프레임은 8번과 12번이다(거리가 같다)
    assert find_impact_frame(pose, ball) in (8, 12)


def test_ignores_frames_without_pose(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    pose[10] = np.nan
    assert find_impact_frame(pose, ball) in (9, 11)


def test_returns_none_when_ball_never_detected(make_clip):
    pose, ball = make_clip()
    ball[:] = np.nan
    assert find_impact_frame(pose, ball) is None


def test_depth_does_not_affect_impact(make_clip):
    """공에는 실제 z가 없으므로 발의 z가 달라져도 결과가 같아야 한다."""
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    pose[10, config.RIGHT_FOOT_INDEX, 2] = 5.0
    assert find_impact_frame(pose, ball) == 10
