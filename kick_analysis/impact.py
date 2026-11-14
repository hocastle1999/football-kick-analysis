"""임팩트(발이 공에 닿는 순간) 추정.

두 가지 방법이 있다.

- find_impact: 지금 쓰는 방법. 멈춰 있는 공의 위치를 먼저 정하고, 차는 발끝이 공의 가장자리를
  지나가는 시점을 프레임 사이까지 계산한다.
- find_impact_frame: 논문의 방법. 발과 공이 가장 가까운 프레임. 비교용으로 남겨 둔다.

방법을 바꾼 이유(docs/evaluation.md):
24fps 영상에서 차는 발은 한 프레임에 화면 폭의 15% 안팎을 움직인다. 실제 접촉은 대부분 두 프레임 사이에
있어서, "가장 가까운 프레임"은 영상에 따라 접촉 직전이 되기도 하고 직후가 되기도 했다.
또 공이 검출된 프레임 안에서만 찾았기 때문에 임팩트 근처에서 공을 놓치면 결과가 밀렸다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kick_analysis import config

LEFT_HIP, RIGHT_HIP = 23, 24


@dataclass(frozen=True)
class Impact:
    frame: int  # 발끝이 아직 공에 닿지 않은 마지막 프레임
    fraction: float  # frame과 frame+1 사이에서 공에 닿는 시점(0 이상 1 미만)
    ball_xy: np.ndarray  # 차기 전 멈춰 있던 공의 중심 (x, y), 화면 기준 0~1
    ball_size: np.ndarray  # 공 상자의 (너비, 높이), 화면 기준 0~1
    direction: int  # 차는 방향. +1이면 화면 오른쪽, -1이면 왼쪽

    @property
    def time(self) -> float:
        """프레임 단위의 접촉 시점. 예: 23.4는 23번과 24번 프레임 사이"""
        return self.frame + self.fraction


def estimate_ball_rest(ball: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """차기 전 멈춰 있는 공의 (중심, 크기)를 추정한다. 공이 한 번도 안 잡혔으면 None.

    공은 차기 전까지 한자리에 있으므로, 검출된 위치 중 가장 많이 모여 있는 곳이 그 자리다.
    날아가는 공이 섞여 있어도 영향을 받지 않도록 평균이 아니라 "이웃이 가장 많은 위치"를 고른다.
    """
    group = _rest_group(ball)
    if len(group) == 0:
        return None
    return group[:, :2].mean(axis=0), np.median(group[:, 2:4], axis=0)


def count_rest_detections(ball: np.ndarray) -> int:
    """공이 같은 자리에서 몇 번 검출됐는지 센다.

    추출기가 "공의 자리를 충분히 확인했으니 공 검출을 그만해도 된다"를 판단할 때 쓴다.
    """
    return len(_rest_group(ball))


def _rest_group(ball: np.ndarray) -> np.ndarray:
    """검출된 공 중 가장 많이 모여 있는 묶음의 행들을 돌려준다. 검출이 없으면 빈 배열."""
    detected = ball[~np.isnan(ball[:, 0])]
    if len(detected) == 0:
        return detected

    centers = detected[:, :2]
    radius = np.median(detected[:, 2]) / 2  # 공 반지름 안에 들어오면 같은 자리로 본다
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    neighbors = distances <= radius
    best = int(neighbors.sum(axis=1).argmax())  # 동점이면 가장 앞 프레임
    return detected[neighbors[best]]


def find_impact(pose: np.ndarray, ball: np.ndarray) -> Impact | None:
    """차는 발끝이 멈춰 있는 공의 가장자리를 지나는 시점을 찾는다.

    pose: (T, 33, 4), ball: (T, 5). 못 찾은 프레임은 NaN.
    공이 한 번도 안 잡혔거나 발이 안 잡혔으면 None.
    """
    rest = estimate_ball_rest(ball)
    if rest is None:
        return None
    ball_xy, ball_size = rest

    toe_x = pose[:, config.RIGHT_FOOT_INDEX, 0]
    hip_x = pose[:, [LEFT_HIP, RIGHT_HIP], 0].mean(axis=1)
    if np.all(np.isnan(toe_x)):
        return None

    # 사람이 처음에 공의 어느 쪽에 있었는지로 차는 방향을 정한다
    start_x = np.nanmedian(hip_x[~np.isnan(hip_x)][:5])
    direction = 1 if ball_xy[0] >= start_x else -1

    # progress: 발끝이 공의 가까운 쪽 가장자리까지 남은 거리. 음수면 아직 못 미침, 양수면 지나감
    near_edge = ball_xy[0] - direction * ball_size[0] / 2
    progress = direction * (toe_x - near_edge)

    # 음수에서 양수로 바뀌는 구간이 접촉이다. 여러 번이면 가장 빠르게 지나간 구간(킥 스윙)을 고른다
    before, after = progress[:-1], progress[1:]
    with np.errstate(invalid="ignore"):
        crossings = np.flatnonzero((before < 0) & (after >= 0))
    if len(crossings) > 0:
        steps = after[crossings] - before[crossings]
        frame = int(crossings[steps.argmax()])
        fraction = float(-progress[frame] / (progress[frame + 1] - progress[frame]))
        return Impact(frame, min(fraction, 0.999), ball_xy, ball_size, direction)

    # 발끝이 공을 지나가지 않는 영상(헛발질 등): 공에 가장 가까이 간 프레임을 쓴다
    toe_xy = pose[:, config.RIGHT_FOOT_INDEX, :2]
    frame = int(np.nanargmin(np.linalg.norm(toe_xy - ball_xy, axis=1)))
    return Impact(frame, 0.0, ball_xy, ball_size, direction)


def find_impact_frame(pose: np.ndarray, ball: np.ndarray) -> int | None:
    """(논문의 방법) 오른발 끝과 공 중심이 화면에서 가장 가까운 프레임 번호를 돌려준다.

    pose: (T, 33, 4), ball: (T, 5). 못 찾은 프레임은 NaN이어야 한다.
    발과 공이 둘 다 잡힌 프레임만 후보로 삼고, 후보가 없으면 None.

    거리는 화면 좌표 (x, y)만으로 잰다. 공은 2D 검출이라 실제 깊이(z) 값이 없기 때문이다.
    """
    foot_xy = pose[:, config.RIGHT_FOOT_INDEX, :2]
    ball_xy = ball[:, :2]

    distance = np.linalg.norm(foot_xy - ball_xy, axis=1)  # 한쪽이라도 NaN이면 결과도 NaN
    if np.all(np.isnan(distance)):
        return None
    return int(np.nanargmin(distance))
