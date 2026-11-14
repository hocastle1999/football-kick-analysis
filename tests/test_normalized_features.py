"""공 기준 정규화 좌표 테스트. 핵심은 "촬영 조건이 달라도 같은 자세면 같은 값"이다."""

import numpy as np
import pytest

from kick_analysis import features
from kick_analysis.impact import Impact

ASPECT = 16 / 9


def make_pose(rng):
    """관절 33개의 임의 자세 2프레임. 어깨와 골반은 몸통 길이가 0이 되지 않게 정한다."""
    pose = np.zeros((2, 33, 4), dtype=np.float64)
    pose[:, :, :2] = rng.uniform(0.3, 0.7, size=(1, 33, 2))
    pose[:, [features.LEFT_SHOULDER, features.RIGHT_SHOULDER], :2] = (0.40, 0.40)
    pose[:, [features.LEFT_HIP, features.RIGHT_HIP], :2] = (0.42, 0.60)
    pose[1, :, 0] += 0.02  # 다음 프레임에서는 전체가 조금 앞으로 간다
    return pose


def make_impact(ball_xy=(0.6, 0.8), direction=1, fraction=0.5):
    return Impact(0, fraction, np.array(ball_xy), np.array([0.05, 0.09]), direction)


def test_pose_at_interpolates_between_frames():
    pose = np.zeros((3, 33, 4))
    pose[1] = 10.0
    pose[2] = 20.0
    np.testing.assert_allclose(features.pose_at(pose, 1.0), 10.0)
    np.testing.assert_allclose(features.pose_at(pose, 1.25), 12.5)
    np.testing.assert_allclose(features.pose_at(pose, 2.0), 20.0)  # 마지막 프레임은 그대로


def test_ball_is_origin_and_torso_is_unit():
    pose = make_pose(np.random.default_rng(0))
    normalized = features.build_normalized_pose(pose, make_impact(), ASPECT)

    assert normalized.shape == (33, 2)
    shoulder = normalized[[features.LEFT_SHOULDER, features.RIGHT_SHOULDER]].mean(axis=0)
    hip = normalized[[features.LEFT_HIP, features.RIGHT_HIP]].mean(axis=0)
    assert np.linalg.norm(shoulder - hip) == pytest.approx(1.0, abs=1e-5)
    assert hip[0] < 0  # 골반이 공보다 뒤(차는 방향의 반대쪽)에 있다


def test_same_pose_regardless_of_position_in_frame():
    pose = make_pose(np.random.default_rng(1))
    base = features.build_normalized_pose(pose, make_impact(), ASPECT)

    shifted = pose.copy()
    shifted[:, :, 0] += 0.1
    shifted[:, :, 1] -= 0.05
    moved = features.build_normalized_pose(shifted, make_impact(ball_xy=(0.7, 0.75)), ASPECT)
    np.testing.assert_allclose(moved, base, atol=1e-5)


def test_same_pose_regardless_of_camera_distance():
    pose = make_pose(np.random.default_rng(2))
    base = features.build_normalized_pose(pose, make_impact(), ASPECT)

    zoomed = pose.copy()
    zoomed[:, :, :2] *= 0.5  # 카메라가 두 배 멀어져 모든 것이 절반 크기로 찍힌 경우
    far = features.build_normalized_pose(zoomed, make_impact(ball_xy=(0.3, 0.4)), ASPECT)
    np.testing.assert_allclose(far, base, atol=1e-5)


def test_same_pose_when_filmed_from_the_other_side():
    pose = make_pose(np.random.default_rng(3))
    base = features.build_normalized_pose(pose, make_impact(), ASPECT)

    mirrored = pose.copy()
    mirrored[:, :, 0] = 1 - mirrored[:, :, 0]
    flipped = features.build_normalized_pose(mirrored, make_impact(ball_xy=(0.4, 0.8), direction=-1), ASPECT)
    np.testing.assert_allclose(flipped, base, atol=1e-5)


def test_none_when_pose_missing_at_impact():
    pose = make_pose(np.random.default_rng(4))
    pose[1] = np.nan
    assert features.build_normalized_pose(pose, make_impact(), ASPECT) is None
    assert features.build_normalized_pose(pose, make_impact(fraction=0.0), ASPECT) is not None


def test_normalized_sequence_shape_and_edges():
    rng = np.random.default_rng(5)
    pose = np.repeat(make_pose(rng)[:1], 12, axis=0)
    impact = Impact(5, 0.3, np.array([0.6, 0.8]), np.array([0.05, 0.09]), 1)

    sequence = features.build_normalized_sequence(pose, impact, ASPECT, before=2, length=6)
    assert sequence.shape == (6, 66)
    # 모든 프레임이 같은 자세이므로 각 행은 접촉 시점의 정규화 자세와 같다
    single = features.build_normalized_pose(pose, impact, ASPECT).reshape(-1)
    np.testing.assert_allclose(sequence[0], single, atol=1e-5)

    early = Impact(1, 0.3, impact.ball_xy, impact.ball_size, 1)
    late = Impact(9, 0.3, impact.ball_xy, impact.ball_size, 1)
    assert features.build_normalized_sequence(pose, early, ASPECT, before=2, length=6) is None
    assert features.build_normalized_sequence(pose, late, ASPECT, before=2, length=6) is None


def test_weight_geometry_signs():
    pose = make_pose(np.random.default_rng(6))
    normalized = features.build_normalized_pose(pose, make_impact(), ASPECT)
    hip_x, shoulder_x, nose_x = features.weight_geometry(normalized)
    assert hip_x == pytest.approx(normalized[[23, 24], 0].mean())
    assert shoulder_x < hip_x  # 이 자세는 어깨가 골반보다 뒤에 있다(뒤로 기운 자세)
    assert nose_x == pytest.approx(normalized[0, 0])
