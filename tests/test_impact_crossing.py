"""새 임팩트 추정(find_impact) 테스트.

make_clip의 가짜 좌표: 공은 (0.5, 0.8)에 멈춰 있고 너비 0.05, 오른발 끝 x는 프레임마다 0.05씩 늘어
impact_frame에서 공 중심(0.5)에 온다. 공의 가까운 쪽 가장자리는 0.475이므로
발끝은 (impact_frame - 1)번과 impact_frame번 프레임의 정확히 중간에서 가장자리를 지난다.
"""

import numpy as np
import pytest

from kick_analysis import config
from kick_analysis.impact import estimate_ball_rest, find_impact


def test_finds_crossing_between_frames(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    result = find_impact(pose, ball)

    assert result.frame == 9  # 아직 닿지 않은 마지막 프레임
    assert result.fraction == pytest.approx(0.5, abs=1e-4)
    assert result.time == pytest.approx(9.5, abs=1e-4)
    assert result.direction == 1
    np.testing.assert_allclose(result.ball_xy, [0.5, 0.8], atol=1e-6)


def test_works_when_ball_is_lost_around_impact(make_clip):
    """임팩트 근처에서 공을 놓쳐도, 차기 전 위치를 알고 있으므로 결과가 같다."""
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    ball[7:] = np.nan
    result = find_impact(pose, ball)
    assert (result.frame, round(result.fraction, 3)) == (9, 0.5)


def test_flying_ball_does_not_move_rest_position(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    for t in range(11, 20):  # 찬 뒤에는 공이 프레임마다 0.08씩 날아간다
        ball[t, 0] = 0.5 + 0.08 * (t - 10)
    center, size = estimate_ball_rest(ball)
    np.testing.assert_allclose(center, [0.5, 0.8], atol=1e-6)
    np.testing.assert_allclose(size, [0.05, 0.05], atol=1e-6)
    assert find_impact(pose, ball).frame == 9


def test_kick_toward_left_side_of_screen(make_clip):
    """화면을 좌우로 뒤집어도 같은 시점을 찾고, 방향만 -1이 된다."""
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    pose[:, :, 0] = 1 - pose[:, :, 0]
    ball[:, 0] = 1 - ball[:, 0]
    result = find_impact(pose, ball)
    assert result.direction == -1
    assert (result.frame, round(result.fraction, 3)) == (9, 0.5)


def test_none_without_ball(make_clip):
    pose, ball = make_clip()
    ball[:] = np.nan
    assert estimate_ball_rest(ball) is None
    assert find_impact(pose, ball) is None


def test_falls_back_to_closest_frame_when_toe_never_reaches_ball(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    # 발끝이 공 앞 0.1 지점(15번 프레임)까지만 갔다가 멈춘다
    pose[:, config.RIGHT_FOOT_INDEX, 0] = np.minimum(pose[:, config.RIGHT_FOOT_INDEX, 0], 0.4)
    result = find_impact(pose, ball)
    assert result.fraction == 0.0
    assert pose[result.frame, config.RIGHT_FOOT_INDEX, 0] == pytest.approx(0.4)


def test_picks_the_fast_swing_when_toe_crosses_twice(make_clip):
    pose, ball = make_clip(num_frames=30, impact_frame=20)
    toe = pose[:, config.RIGHT_FOOT_INDEX, 0]
    # 3~5번 프레임에서 발끝이 가장자리(0.475)를 살짝 넘었다가 돌아온다(검출 흔들림을 흉내 낸다)
    toe[3:6] = [0.47, 0.48, 0.47]
    result = find_impact(pose, ball)
    assert result.frame == 19
