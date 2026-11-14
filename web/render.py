"""임팩트 순간의 프레임에 관절과 공을 그려 JPEG로 만든다."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kick_analysis.extraction import VideoKeypoints
from kick_analysis.impact import estimate_ball_rest

# BlazePose 관절 번호를 잇는 선(몸통, 팔, 다리, 발). 얼굴은 그리지 않는다
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),  # 팔
    (23, 25), (25, 27), (24, 26), (26, 28),  # 다리
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),  # 발
]  # fmt: skip
MAX_WIDTH = 1280  # 이보다 큰 영상은 줄여서 보낸다


def render_frame(video_path: Path, keypoints: VideoKeypoints, frame_number: int) -> bytes:
    """frame_number번째 프레임에 관절 선과 공 상자를 그린 JPEG 바이트를 돌려준다."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"{frame_number}번 프레임을 읽지 못했습니다.")

    height, width = frame.shape[:2]
    thickness = max(2, width // 400)

    pose = keypoints.pose[frame_number]
    if not np.isnan(pose[0, 0]):
        points = [(int(x * width), int(y * height)) for x, y in pose[:, :2]]
        for a, b in SKELETON:
            cv2.line(frame, points[a], points[b], (0, 255, 255), thickness, cv2.LINE_AA)
        for index in range(11, 33):
            cv2.circle(frame, points[index], thickness + 1, (0, 128, 255), -1, cv2.LINE_AA)

    # 공 검출은 공의 자리를 확인하면 멈추므로 이 프레임에는 검출값이 없을 수 있다.
    # 그럴 때는 차기 전 공의 자리를 그린다(접촉 순간에는 공이 아직 그 자리에 있다).
    ball = keypoints.ball[frame_number, :4]
    if np.isnan(ball[0]):
        rest = estimate_ball_rest(keypoints.ball)
        ball = None if rest is None else np.concatenate(rest)
    if ball is not None:
        cx, cy, w, h = ball[0] * width, ball[1] * height, ball[2] * width, ball[3] * height
        top_left = (int(cx - w / 2), int(cy - h / 2))
        bottom_right = (int(cx + w / 2), int(cy + h / 2))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), thickness)

    if width > MAX_WIDTH:
        frame = cv2.resize(frame, (MAX_WIDTH, int(height * MAX_WIDTH / width)))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError("이미지를 만들지 못했습니다.")
    return encoded.tobytes()
