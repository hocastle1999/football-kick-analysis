"""프레임별 좌표를 LSTM 입력 시퀀스로 바꾼다.

학습(dataset.py)과 추론(GUI)이 같은 함수를 쓰므로 두 쪽의 입력 형태가 항상 같다.
못 찾은 값(NaN)은 0으로 채운다.
"""

from __future__ import annotations

import numpy as np

from kick_analysis import config


def build_impact_sequence(pose: np.ndarray, impact_frame: int) -> np.ndarray | None:
    """무게중심 모델 입력 (6, 99): 임팩트 앞 2프레임부터 6프레임의 관절 x, y, z.

    임팩트가 영상 맨 앞이면 0번 프레임부터 6프레임을 쓴다.
    뒤쪽 프레임이 모자라면 None.
    """
    start = max(0, impact_frame - config.IMPACT_SEQ_BEFORE)
    end = start + config.IMPACT_SEQ_LEN
    if end > len(pose):
        return None

    clip = np.nan_to_num(pose[start:end, :, :3], nan=0.0)  # (6, 33, 3)
    return clip.reshape(config.IMPACT_SEQ_LEN, -1).astype(np.float32)


def build_foot_sequence(pose: np.ndarray, ball: np.ndarray, impact_frame: int) -> np.ndarray | None:
    """디딤발 모델 입력 (11, 305): 임팩트 앞뒤 5프레임의 관절과 공, 그리고 추가 값 2개.

    한 프레임의 구성: 랜드마크 101개 x 3 (관절 0~32, 공 100, 나머지는 0) + [발-공 거리, y 차이].
    추가 값 2개는 임팩트 프레임에서 한 번 계산해 11프레임에 똑같이 붙인다.
    앞이나 뒤 프레임이 모자라면 None.
    """
    start = impact_frame - config.FOOT_SEQ_BEFORE
    end = start + config.FOOT_SEQ_LEN
    if start < 0 or end > len(pose):
        return None

    # 공은 2D로만 검출되므로, 임팩트 때 오른발의 z를 공의 z로 쓴다(논문의 보정 방식)
    foot_at_impact = np.nan_to_num(pose[impact_frame, config.RIGHT_FOOT_INDEX, :3], nan=0.0)
    ball_z = foot_at_impact[2]

    landmarks = np.zeros((config.FOOT_SEQ_LEN, config.FOOT_NUM_LANDMARKS, 3), dtype=np.float32)
    landmarks[:, : config.NUM_POSE_LANDMARKS] = np.nan_to_num(pose[start:end, :, :3], nan=0.0)

    ball_clip = ball[start:end]
    detected = ~np.isnan(ball_clip[:, 0])
    landmarks[detected, config.FOOT_BALL_INDEX, :2] = ball_clip[detected, :2]
    landmarks[detected, config.FOOT_BALL_INDEX, 2] = ball_z

    extra = np.zeros(config.FOOT_EXTRA_FEATURES, dtype=np.float32)
    if not np.isnan(ball[impact_frame, 0]):
        ball_at_impact = np.array([ball[impact_frame, 0], ball[impact_frame, 1], ball_z])
        extra[0] = np.linalg.norm(foot_at_impact - ball_at_impact)
        extra[1] = foot_at_impact[1] - ball_at_impact[1]

    flat = landmarks.reshape(config.FOOT_SEQ_LEN, -1)
    return np.concatenate([flat, np.tile(extra, (config.FOOT_SEQ_LEN, 1))], axis=1)


# ---------------------------------------------------------------------------
# 공 기준으로 정규화한 좌표
#
# 위의 두 함수는 화면 좌표를 그대로 쓴다. 그러면 같은 자세라도 사람이 화면 어디에 서 있는지,
# 카메라가 얼마나 멀리 있는지에 따라 값이 달라진다. 아래 함수들은 좌표를 이렇게 바꾼다.
#   1. 원점을 차기 전 공의 중심으로 옮긴다          → 화면 속 위치에 영향을 받지 않는다
#   2. 차는 방향이 항상 +x가 되도록 뒤집는다         → 왼쪽에서 찍든 오른쪽에서 찍든 같다
#   3. 몸통 길이(어깨 중심~골반 중심)로 나눈다       → 카메라 거리와 해상도에 영향을 받지 않는다
# ---------------------------------------------------------------------------

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
NOSE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT = 0, 27, 29, 31


def pose_at(pose: np.ndarray, time: float) -> np.ndarray:
    """프레임 사이 시점의 관절 좌표 (33, 4)를 앞뒤 프레임에서 선형 보간한다.

    예: time=23.4면 23번 프레임 60% + 24번 프레임 40%.
    """
    frame = int(np.floor(time))
    fraction = time - frame
    if fraction == 0 or frame + 1 >= len(pose):
        return pose[frame]
    return pose[frame] * (1 - fraction) + pose[frame + 1] * fraction


def _to_ball_frame(points_xy: np.ndarray, impact, aspect: float) -> np.ndarray:
    """화면 좌표 (..., 2)를 공이 원점이고 차는 방향이 +x인 좌표로 바꾼다. 단위는 화면 높이."""
    # x는 화면 너비, y는 화면 높이로 나눈 값이라 눈금이 다르다. aspect(너비/높이)를 곱해 맞춘다
    x = (points_xy[..., 0] - impact.ball_xy[0]) * aspect * impact.direction
    y = points_xy[..., 1] - impact.ball_xy[1]
    return np.stack([x, y], axis=-1)


def _torso_length(points: np.ndarray) -> float:
    shoulder = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
    hip = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    return float(np.linalg.norm(shoulder - hip))


def build_normalized_pose(pose: np.ndarray, impact, aspect: float) -> np.ndarray | None:
    """접촉 시점의 관절 33개를 공 기준 좌표로 바꾼 (33, 2). 그 시점에 관절이 안 잡혔으면 None."""
    points = _to_ball_frame(pose_at(pose, impact.time)[:, :2], impact, aspect)
    if np.isnan(points).any():
        return None
    return (points / _torso_length(points)).astype(np.float32)


def build_normalized_sequence(
    pose: np.ndarray, impact, aspect: float, before: int, length: int
) -> np.ndarray | None:
    """접촉 직전 프레임을 기준으로 앞 before프레임부터 length프레임의 정규화 좌표 (length, 66).

    모든 프레임을 접촉 시점의 몸통 길이 하나로 나눈다(프레임마다 다른 값으로 나누면 움직임이 왜곡된다).
    앞뒤 프레임이 모자라거나 접촉 시점에 관절이 안 잡혔으면 None.
    """
    start = impact.frame - before
    end = start + length
    if start < 0 or end > len(pose):
        return None
    at_impact = _to_ball_frame(pose_at(pose, impact.time)[:, :2], impact, aspect)
    if np.isnan(at_impact).any():
        return None

    clip = _to_ball_frame(pose[start:end, :, :2], impact, aspect) / _torso_length(at_impact)
    return np.nan_to_num(clip, nan=0.0).reshape(length, -1).astype(np.float32)


def weight_geometry(normalized_pose: np.ndarray) -> np.ndarray:
    """무게중심 판단에 쓰는 값 3개: 골반, 어깨, 코가 공보다 얼마나 앞(+)이나 뒤(-)에 있는지."""
    hip = (normalized_pose[LEFT_HIP] + normalized_pose[RIGHT_HIP]) / 2
    shoulder = (normalized_pose[LEFT_SHOULDER] + normalized_pose[RIGHT_SHOULDER]) / 2
    return np.array([hip[0], shoulder[0], normalized_pose[NOSE, 0]], dtype=np.float32)


def foot_geometry(normalized_pose: np.ndarray) -> np.ndarray:
    """디딤발 판단에 쓰는 값 6개: 왼발의 발목, 뒤꿈치, 발끝이 공 기준으로 어디 있는지 (x, y)."""
    return normalized_pose[[LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT]].reshape(-1).astype(np.float32)


# ---- 지금 쓰는 분류기(로지스틱 회귀)의 입력. 학습과 분석이 이 두 함수를 같이 쓴다 ----


def build_weight_features(pose: np.ndarray, impact, aspect: float) -> np.ndarray | None:
    """무게중심 분류기 입력 (66,): 접촉 시점의 관절 33개를 공 기준으로 정규화한 (x, y)."""
    normalized = build_normalized_pose(pose, impact, aspect)
    return None if normalized is None else normalized.reshape(-1)


def build_foot_features(pose: np.ndarray, impact, aspect: float) -> np.ndarray | None:
    """디딤발 분류기 입력 (6,): 접촉 시점의 왼발 발목, 뒤꿈치, 발끝 위치(공 기준).

    관절 전체가 아니라 디딤발 위치만 쓴다. 사람 단위 평가에서 이쪽이 더 잘 맞았고,
    디딤발 라벨은 "발이 공 기준으로 어디 있는가"만으로 정해지기 때문이다.
    """
    normalized = build_normalized_pose(pose, impact, aspect)
    return None if normalized is None else foot_geometry(normalized)
