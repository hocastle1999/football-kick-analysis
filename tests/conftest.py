import numpy as np
import pytest

from kick_analysis import config


@pytest.fixture
def make_clip():
    """발이 공으로 다가갔다가 지나가는 가짜 좌표를 만든다.

    공은 화면 (0.5, 0.8)에 멈춰 있고, 오른발 x가 프레임마다 0.05씩 움직여
    impact_frame에서 공과 겹친다.
    """

    def _make(num_frames: int = 20, impact_frame: int = 10):
        pose = np.zeros((num_frames, config.NUM_POSE_LANDMARKS, 4), dtype=np.float32)
        pose[:, :, :3] = 0.1  # 나머지 관절은 임의의 고정값
        pose[:, :, 3] = 1.0
        for t in range(num_frames):
            pose[t, config.RIGHT_FOOT_INDEX, :3] = (0.5 + 0.05 * (t - impact_frame), 0.8, -0.2)

        ball = np.zeros((num_frames, 5), dtype=np.float32)
        ball[:] = (0.5, 0.8, 0.05, 0.05, 0.9)
        return pose, ball

    return _make


@pytest.fixture
def make_body_clip(make_clip):
    """make_clip에 어깨와 골반 위치를 넣은 것. 몸통 길이가 있어야 정규화 좌표를 만들 수 있다."""

    def _make(num_frames: int = 20, impact_frame: int = 10, lean: float = 0.0):
        pose, ball = make_clip(num_frames, impact_frame)
        pose[:, [23, 24], :2] = (0.42, 0.60)  # 골반
        pose[:, [11, 12], :2] = (0.42 + lean, 0.40)  # 어깨. lean이 음수면 뒤로 기운 자세
        return pose, ball

    return _make
