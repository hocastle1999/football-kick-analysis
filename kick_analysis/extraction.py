"""영상에서 프레임별 관절 좌표와 공 위치를 뽑는다.

학습용 데이터 생성과 GUI 분석이 모두 이 모듈을 쓴다.
예전에는 두 곳이 각자 다른 설정으로 좌표를 뽑아서 학습과 추론의 입력이 달랐다.

핵심 규칙: 결과 배열의 i번째 행은 항상 영상의 i번째 프레임이다.
관절이나 공을 못 찾은 프레임도 지우지 않고 NaN으로 남긴다.
(예전 코드는 관절을 못 찾은 프레임을 지워서, 그 뒤로 관절과 공이 서로 다른 프레임의 값이 되었다.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kick_analysis import config
from kick_analysis.impact import count_rest_detections, find_impact


@dataclass
class VideoKeypoints:
    """영상 하나에서 뽑은 프레임별 좌표. T는 프레임 수."""

    pose: np.ndarray  # (T, 33, 4) x, y, z, visibility. x와 y는 화면 크기 기준 0~1
    pose_world: np.ndarray  # (T, 33, 3) 골반 중심이 원점인 미터 단위 좌표
    ball: np.ndarray  # (T, 5) 중심 x, 중심 y, 너비, 높이(모두 0~1), 신뢰도
    fps: float
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return len(self.pose)

    @property
    def pose_detected(self) -> np.ndarray:
        """프레임별 관절 검출 여부 (T,)"""
        return ~np.isnan(self.pose[:, 0, 0])

    @property
    def ball_detected(self) -> np.ndarray:
        """프레임별 공 검출 여부 (T,)"""
        return ~np.isnan(self.ball[:, 0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pose=self.pose.astype(np.float32),
            pose_world=self.pose_world.astype(np.float32),
            ball=self.ball.astype(np.float32),
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    @classmethod
    def load(cls, path: str | Path) -> VideoKeypoints:
        with np.load(path) as data:
            return cls(
                pose=data["pose"],
                pose_world=data["pose_world"],
                ball=data["ball"],
                fps=float(data["fps"]),
                width=int(data["width"]),
                height=int(data["height"]),
            )


class KeypointExtractor:
    """MediaPipe BlazePose(관절)와 YOLOv8n(공)을 묶은 추출기.

    모델을 불러오는 데 시간이 걸리므로 한 번 만들어 여러 영상에 다시 쓴다.

    관절은 두 단계로 뽑는다(docs/performance.md).
      1단계: 빠른 포즈 모델로 영상 전체를 훑어 임팩트가 어디쯤인지 찾는다.
      2단계: 임팩트 주변 몇 프레임만 정밀한 포즈 모델로 다시 뽑아 그 구간을 바꿔 넣는다.
    분류에 쓰는 것은 임팩트 순간의 자세뿐이라, 나머지 프레임까지 정밀하게 볼 필요가 없다.
    fast_complexity=None이면 처음부터 정밀한 모델 하나로 전체를 뽑는다.
    """

    def __init__(
        self,
        model_complexity: int = config.POSE_MODEL_COMPLEXITY,
        fast_complexity: int | None = config.POSE_FAST_COMPLEXITY,
        yolo_weights: str = config.YOLO_WEIGHTS,
        ball_stop_after: int | None = config.BALL_STOP_AFTER,
    ) -> None:
        # 무거운 라이브러리는 실제로 추출기를 만들 때만 불러온다(테스트는 이 클래스 없이도 돈다)
        import mediapipe as mp
        from ultralytics import YOLO

        self._mp_pose = mp.solutions.pose
        self._model_complexity = model_complexity  # 정밀한 모델
        self._fast_complexity = fast_complexity  # 전체를 훑는 빠른 모델. None이면 쓰지 않는다
        self._yolo = YOLO(yolo_weights)
        self._ball_stop_after = ball_stop_after  # None이면 모든 프레임에서 공을 찾는다

    def extract(
        self,
        video_path: str | Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> VideoKeypoints:
        """영상 전체를 읽어 프레임별 좌표를 돌려준다.

        on_progress(처리한 양, 전체 양)을 주면 프레임마다 호출한다.
        """
        two_pass = self._fast_complexity is not None
        first_complexity = self._fast_complexity if two_pass else self._model_complexity
        refine_frames = config.REFINE_BEFORE + config.REFINE_AFTER + 1 if two_pass else 0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        poses, worlds, balls = [], [], []
        no_ball = np.full(5, np.nan, dtype=np.float32)
        searching_ball = True
        # static_image_mode=False: 앞 프레임의 결과를 이어받아 추적한다(영상용 설정).
        # 추적 상태가 다음 영상으로 넘어가지 않도록 영상마다 새로 만든다.
        with self._mp_pose.Pose(static_image_mode=False, model_complexity=first_complexity) as pose_model:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pose, world = self._detect_pose(pose_model, frame)
                poses.append(pose)
                worlds.append(world)

                if searching_ball:
                    ball = self._detect_ball(frame)
                    balls.append(ball)
                    # 공은 차기 전까지 한자리에 있다. 같은 자리에서 충분히 잡혔으면 자리를 안 것이므로
                    # 남은 프레임에서는 공 검출을 건너뛴다(공 검출이 프레임당 약 100ms로 비싸다).
                    if (
                        self._ball_stop_after
                        and not np.isnan(ball[0])
                        and count_rest_detections(np.stack(balls)) >= self._ball_stop_after
                    ):
                        searching_ball = False
                else:
                    balls.append(no_ball)

                if on_progress:
                    on_progress(len(poses), total + refine_frames)
        cap.release()

        if not poses:
            raise ValueError(f"프레임을 하나도 읽지 못했습니다: {video_path}")

        keypoints = VideoKeypoints(
            pose=np.stack(poses),
            pose_world=np.stack(worlds),
            ball=np.stack(balls),
            fps=float(fps),
            width=width,
            height=height,
        )
        if two_pass:
            self.refine_around_impact(video_path, keypoints)
            if on_progress:  # 2단계까지 끝났음을 알린다
                on_progress(len(poses) + refine_frames, len(poses) + refine_frames)
        return keypoints

    def refine_around_impact(self, video_path: str | Path, keypoints: VideoKeypoints) -> tuple[int, int]:
        """임팩트 주변 구간의 관절을 정밀한 모델로 다시 뽑아 keypoints에 바꿔 넣는다.

        다시 뽑은 구간 (시작, 끝)을 돌려준다. 끝 프레임은 포함하지 않는다.
        임팩트를 찾지 못하면 아무것도 하지 않고 (0, 0)을 돌려준다.
        """
        found = find_impact(keypoints.pose, keypoints.ball)
        if found is None:
            return 0, 0
        start = max(0, found.frame - config.REFINE_BEFORE)
        end = min(keypoints.num_frames, found.frame + config.REFINE_AFTER + 1)
        self._replace_pose(video_path, keypoints, start, end)

        # 정밀한 좌표로 임팩트를 다시 찾는다. 분류는 접촉 직전과 직후 두 프레임의 자세를 쓰므로
        # 두 프레임이 모두 다시 뽑은 구간 안에 있어야 한다.
        # 빠른 모델이 임팩트를 크게 잘못 짚어 구간을 벗어났다면 전체를 정밀한 모델로 다시 뽑는다.
        again = find_impact(keypoints.pose, keypoints.ball)
        if again is None or not (start <= again.frame and again.frame + 1 < end):
            start, end = 0, keypoints.num_frames
            self._replace_pose(video_path, keypoints, start, end)
        return start, end

    def _replace_pose(self, video_path: str | Path, keypoints: VideoKeypoints, start: int, end: int) -> None:
        """영상의 start~end 프레임을 정밀한 모델로 처리해 keypoints의 그 구간을 덮어쓴다."""
        cap = cv2.VideoCapture(str(video_path))
        for _ in range(start):
            cap.grab()  # 앞부분은 화면으로 풀지 않고 넘기기만 한다
        with self._mp_pose.Pose(
            static_image_mode=False, model_complexity=self._model_complexity
        ) as pose_model:
            for index in range(start, end):
                ok, frame = cap.read()
                if not ok:
                    break
                keypoints.pose[index], keypoints.pose_world[index] = self._detect_pose(pose_model, frame)
        cap.release()

    def _detect_pose(self, pose_model, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """프레임 하나의 관절 좌표. 못 찾으면 전부 NaN."""
        pose = np.full((config.NUM_POSE_LANDMARKS, 4), np.nan, dtype=np.float32)
        world = np.full((config.NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)

        result = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                pose[i] = (lm.x, lm.y, lm.z, lm.visibility)
        if result.pose_world_landmarks:
            for i, lm in enumerate(result.pose_world_landmarks.landmark):
                world[i] = (lm.x, lm.y, lm.z)
        return pose, world

    def _detect_ball(self, frame: np.ndarray) -> np.ndarray:
        """프레임 하나의 공 위치. 여러 개가 잡히면 신뢰도가 가장 높은 것, 없으면 NaN."""
        ball = np.full(5, np.nan, dtype=np.float32)

        result = self._yolo.predict(frame, classes=[config.BALL_CLASS_ID], verbose=False)[0]
        if len(result.boxes) > 0:
            best = int(result.boxes.conf.argmax())
            # xywhn: 화면 크기로 나눈 (중심 x, 중심 y, 너비, 높이)
            ball[:4] = result.boxes.xywhn[best].cpu().numpy()
            ball[4] = float(result.boxes.conf[best])
        return ball
