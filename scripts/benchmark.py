"""좌표 추출 시간을 잰다. 결과는 docs/performance.md에 정리한다.

사용 예 (프로젝트 루트에서):
    python -m scripts.benchmark stages "영상1.mp4" "영상2.MOV"
    python -m scripts.benchmark configs "영상1.mp4" "영상2.MOV" --repeats 3

stages:  영상 읽기, 포즈 추정(모델 크기 0, 1, 2), 공 검출을 따로 돌려 프레임당 시간을 낸다.
         무엇을 줄여야 전체가 빨라지는지 알기 위한 것이다.
configs: 추출기 설정별로 영상 하나를 처음부터 끝까지 처리하는 시간을 잰다.
         노트북은 발열에 따라 속도가 달라지므로 설정을 번갈아 여러 번 돌려 가운데 값을 쓴다.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

from kick_analysis import config
from kick_analysis.extraction import KeypointExtractor

# 이름: (정밀 모델 크기, 빠른 모델 크기, 공 검출 조기 종료)
CONFIGS = {
    "처음 (크기 2, 공 매 프레임)": (2, None, None),
    "공 검출 조기 종료": (2, None, config.BALL_STOP_AFTER),
    "2단계 (크기 0 → 임팩트 주변만 크기 2)": (2, 0, config.BALL_STOP_AFTER),
    "크기 0만": (0, None, config.BALL_STOP_AFTER),
}


def read_frames(path: Path) -> tuple[list, float]:
    """영상을 전부 읽어 (프레임 목록, 걸린 시간)을 돌려준다."""
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, time.perf_counter() - started


def time_pose(frames: list, complexity: int) -> float:
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=complexity) as pose:
        pose.process(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))  # 첫 호출의 준비 시간은 빼고 잰다
        started = time.perf_counter()
        for frame in frames:
            pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return time.perf_counter() - started


def time_ball(frames: list, yolo: YOLO) -> float:
    yolo.predict(frames[0], classes=[config.BALL_CLASS_ID], verbose=False)
    started = time.perf_counter()
    for frame in frames:
        yolo.predict(frame, classes=[config.BALL_CLASS_ID], verbose=False)
    return time.perf_counter() - started


def run_stages(videos: list[Path]) -> None:
    yolo = YOLO(config.YOLO_WEIGHTS)
    print("| 영상 | 프레임 | 읽기 | 포즈 0 | 포즈 1 | 포즈 2 | 공 검출 |")
    print("|---|---|---|---|---|---|---|")
    for video in videos:
        frames, read_seconds = read_frames(video)
        n = len(frames)
        cells = [f"{read_seconds / n * 1000:.0f}ms"]
        cells += [f"{time_pose(frames, c) / n * 1000:.0f}ms" for c in (0, 1, 2)]
        cells.append(f"{time_ball(frames, yolo) / n * 1000:.0f}ms")
        height, width = frames[0].shape[:2]
        print(f"| {video.name} ({width}×{height}) | {n} | " + " | ".join(cells) + " |", flush=True)


def run_configs(videos: list[Path], repeats: int) -> None:
    extractors = {
        name: KeypointExtractor(model_complexity=precise, fast_complexity=fast, ball_stop_after=stop)
        for name, (precise, fast, stop) in CONFIGS.items()
    }
    for video in videos:
        seconds = {name: [] for name in extractors}
        frames = 0
        for _ in range(repeats):
            for name, extractor in extractors.items():  # 설정을 번갈아 돌린다
                started = time.perf_counter()
                frames = extractor.extract(video).num_frames
                seconds[name].append(time.perf_counter() - started)

        fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
        print(f"\n{video.name}: {frames}프레임, 영상 길이 {frames / fps:.1f}초, {repeats}회 중 가운데 값")
        print("| 설정 | 전체 시간 | 프레임당 | 처음 대비 |")
        print("|---|---|---|---|")
        baseline = statistics.median(next(iter(seconds.values())))
        for name, values in seconds.items():
            median = statistics.median(values)
            print(
                f"| {name} | {median:.1f}초 | {median / frames * 1000:.0f}ms | {baseline / median:.1f}배 |",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="좌표 추출 시간 측정")
    parser.add_argument("mode", choices=["stages", "configs"])
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "stages":
        run_stages(args.videos)
    else:
        run_configs(args.videos, args.repeats)


if __name__ == "__main__":
    main()
