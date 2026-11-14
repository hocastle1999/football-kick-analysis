import numpy as np

from kick_analysis import config
from kick_analysis.features import build_foot_sequence, build_impact_sequence


def test_impact_sequence_shape_and_window(make_clip):
    pose, _ = make_clip(num_frames=20, impact_frame=10)
    sequence = build_impact_sequence(pose, impact_frame=10)

    assert sequence.shape == (config.IMPACT_SEQ_LEN, config.IMPACT_INPUT_SIZE)
    # 첫 행은 임팩트 2프레임 앞(8번 프레임)의 관절 x, y, z
    np.testing.assert_allclose(sequence[0], pose[8, :, :3].reshape(-1))
    np.testing.assert_allclose(sequence[-1], pose[13, :, :3].reshape(-1))


def test_impact_sequence_at_video_start(make_clip):
    pose, _ = make_clip(num_frames=20, impact_frame=1)
    sequence = build_impact_sequence(pose, impact_frame=1)
    np.testing.assert_allclose(sequence[0], pose[0, :, :3].reshape(-1))


def test_impact_sequence_none_when_too_few_frames_after(make_clip):
    pose, _ = make_clip(num_frames=20, impact_frame=18)
    assert build_impact_sequence(pose, impact_frame=18) is None


def test_impact_sequence_fills_missing_pose_with_zero(make_clip):
    pose, _ = make_clip(num_frames=20, impact_frame=10)
    pose[9] = np.nan
    sequence = build_impact_sequence(pose, impact_frame=10)
    assert not np.isnan(sequence).any()
    assert np.all(sequence[1] == 0)


def test_foot_sequence_layout(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    sequence = build_foot_sequence(pose, ball, impact_frame=10)
    assert sequence.shape == (config.FOOT_SEQ_LEN, config.FOOT_INPUT_SIZE)

    landmarks = sequence[:, : config.FOOT_NUM_LANDMARKS * 3].reshape(config.FOOT_SEQ_LEN, -1, 3)
    # 관절 0~32는 포즈 그대로
    np.testing.assert_allclose(landmarks[:, :33], pose[5:16, :, :3])
    # 33~99는 비어 있다
    assert np.all(landmarks[:, 33:100] == 0)
    # 100번은 공: x, y는 검출값, z는 임팩트 때 오른발의 z
    np.testing.assert_allclose(landmarks[:, 100], np.tile([0.5, 0.8, -0.2], (11, 1)), rtol=1e-6)


def test_foot_sequence_extra_features(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    pose[10, config.RIGHT_FOOT_INDEX, :2] = (0.53, 0.76)  # 공에서 x로 0.03, y로 -0.04
    sequence = build_foot_sequence(pose, ball, impact_frame=10)

    extra = sequence[:, -config.FOOT_EXTRA_FEATURES :]
    np.testing.assert_allclose(extra[:, 0], 0.05, atol=1e-6)  # 거리 = sqrt(0.03² + 0.04²)
    np.testing.assert_allclose(extra[:, 1], -0.04, atol=1e-6)  # 발 y - 공 y


def test_foot_sequence_none_near_video_edges(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    assert build_foot_sequence(pose, ball, impact_frame=4) is None  # 앞 5프레임이 없다
    assert build_foot_sequence(pose, ball, impact_frame=15) is None  # 뒤 5프레임이 없다


def test_foot_sequence_ball_missing_in_some_frames(make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    ball[7] = np.nan
    sequence = build_foot_sequence(pose, ball, impact_frame=10)
    landmarks = sequence[:, : config.FOOT_NUM_LANDMARKS * 3].reshape(config.FOOT_SEQ_LEN, -1, 3)
    assert np.all(landmarks[2, 100] == 0)  # 7번 프레임 = 시퀀스의 2번째 행
    assert not np.isnan(sequence).any()
