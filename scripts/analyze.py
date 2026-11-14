"""영상 하나를 명령줄에서 분석한다 (화면 없이 분석 흐름을 확인할 때 쓴다).

사용 예 (프로젝트 루트에서):
    python -m scripts.analyze "D:/kick_videos/kick_video(mid_18).mp4" --kick long
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kick_analysis.pipeline import RECOMMENDED_WEIGHT, KickAnalyzer, make_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description="킥 영상 분석")
    parser.add_argument("video", type=Path)
    parser.add_argument("--kick", choices=RECOMMENDED_WEIGHT, required=True, help="킥 종류")
    parser.add_argument("--weights", type=Path, default=Path("models/weights"))
    args = parser.parse_args()

    analyzer = KickAnalyzer(args.weights / "weight_linear.npz", args.weights / "foot_linear.npz")
    result = analyzer.analyze(args.video)

    def show(label, confidence):
        return "판단 불가" if label is None else f"{label} (확률 {confidence:.0%})"

    frames = result.keypoints.num_frames
    print(f"영상: {args.video.name} ({frames}프레임, {result.keypoints.fps:.0f}fps)")
    print(f"임팩트: {result.impact_time:.2f}번째 프레임")
    print(f"무게중심: {show(result.weight_label, result.weight_confidence)}")
    print(f"디딤발: {show(result.foot_label, result.foot_confidence)}")
    for line in make_feedback(args.kick, result.weight_label, result.foot_label):
        print(f"  {line}")
    t = result.timings
    print(f"좌표 추출: {t['extract']:.2f}초 (프레임당 {t['extract'] / frames * 1000:.0f}ms)")
    print(f"임팩트 추정과 분류: {t['classify'] * 1000:.2f}ms")
    print(f"전체: {t['total']:.2f}초")


if __name__ == "__main__":
    main()
