"""영상 폴더에서 키포인트를 뽑아 .npz로 저장하고 목록(manifest.csv)을 만든다.

사용 예 (프로젝트 루트에서):
    python -m scripts.extract_keypoints --videos "D:/kick_videos/kick_videos" --source thesis63
    python -m scripts.extract_keypoints --videos "D:/kick_videos/패스슛롱킥" --source extra60
    python -m scripts.extract_keypoints --manifest-only

이미 뽑은 영상은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 한다.
manifest.csv는 저장된 .npz 전체를 다시 읽어 만들기 때문에, 여러 폴더를 따로 실행해도 된다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kick_analysis import dataset, impact
from kick_analysis.extraction import KeypointExtractor, VideoKeypoints

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}


def extract_folder(video_dir: Path, out_dir: Path, source: str) -> None:
    videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f"영상이 없습니다: {video_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = KeypointExtractor()
    for i, video in enumerate(videos, start=1):
        # 결과는 <라벨>/<source>__<영상 이름>.npz 로 저장한다.
        # source를 붙이는 이유는 영상 묶음이 달라도 이름이 겹치지 않게 하려는 것이다
        label = dataset.parse_video_name(video).label
        out_path = out_dir / label / f"{source}__{video.stem}.npz"
        if out_path.exists():
            continue
        started = time.perf_counter()
        keypoints = extractor.extract(video)
        keypoints.save(out_path)
        elapsed = time.perf_counter() - started
        print(
            f"[{i}/{len(videos)}] {video.name}: {keypoints.num_frames}프레임, "
            f"{elapsed:.1f}초 ({elapsed / keypoints.num_frames * 1000:.0f}ms/프레임), "
            f"관절 {keypoints.pose_detected.mean():.0%}, 공 {keypoints.ball_detected.mean():.0%}",
            flush=True,
        )


def build_manifest(out_dir: Path) -> None:
    rows = []
    for npz in sorted(out_dir.rglob("*.npz")):
        source = npz.stem.partition("__")[0]
        info = dataset.parse_video_name(npz)
        keypoints = VideoKeypoints.load(npz)
        found = impact.find_impact(keypoints.pose, keypoints.ball)
        rows.append(
            {
                "file": npz.relative_to(out_dir).as_posix(),
                "label": info.label,
                "kick_type": info.kick_type,
                "clip_id": info.clip_id,
                "source": source,
                "frames": keypoints.num_frames,
                "fps": round(keypoints.fps, 2),
                "width": keypoints.width,
                "height": keypoints.height,
                "pose_rate": round(float(keypoints.pose_detected.mean()), 3),
                # 공 검출은 자리를 확인하면 멈추므로 이 값은 "검출을 시도한 비율"에 가깝다
                "ball_rate": round(float(keypoints.ball_detected.mean()), 3),
                # 프레임 단위의 접촉 시점(예: 22.55는 22번과 23번 프레임 사이)
                "impact_frame": "" if found is None else round(found.time, 2),
            }
        )
    path = dataset.write_manifest(out_dir, rows)
    print(f"목록 저장: {path} ({len(rows)}개)")


def main() -> None:
    parser = argparse.ArgumentParser(description="영상에서 관절과 공 좌표를 추출한다")
    parser.add_argument("--videos", type=Path, help="영상이 들어 있는 폴더")
    parser.add_argument("--out", type=Path, default=Path("data/keypoints/weight"), help="결과 폴더")
    parser.add_argument("--source", default="default", help="이 영상 묶음의 이름(manifest의 source 열)")
    parser.add_argument("--manifest-only", action="store_true", help="추출 없이 목록만 다시 만든다")
    args = parser.parse_args()

    if not args.manifest_only:
        if args.videos is None:
            parser.error("--videos 또는 --manifest-only 중 하나가 필요합니다")
        extract_folder(args.videos, args.out, args.source)
    build_manifest(args.out)


if __name__ == "__main__":
    main()
