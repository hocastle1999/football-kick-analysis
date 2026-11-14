import numpy as np
import pytest

from kick_analysis import config, dataset
from kick_analysis.extraction import VideoKeypoints


@pytest.mark.parametrize(
    ("name", "label", "kick_type", "clip_id"),
    [
        ("kick_video(back_1).mp4", "back", "", 1),
        ("kick_video(mid_15).mp4", "middle", "", 15),
        ("kick_video(front_21).mp4", "front", "", 21),
        ("kick_video_long(back_22).MOV", "back", "long", 22),
        ("kick_video_shoot(front_41).MOV", "front", "shoot", 41),
        ("kick_video_short_pass(middle_30).MOV", "middle", "short_pass", 30),
        ("back/thesis63__kick_video(back_1).npz", "back", "", 1),
    ],
)
def test_parse_weight_video_name(name, label, kick_type, clip_id):
    info = dataset.parse_video_name(name)
    assert (info.label, info.kick_type, info.clip_id) == (label, kick_type, clip_id)


def test_parse_falls_back_to_folder_name():
    info = dataset.parse_video_name("data/foot_videos/too_close/clip3.mp4")
    assert (info.label, info.kick_type, info.clip_id) == ("too_close", "", -1)


def _save_clip(path, pose, ball):
    world = np.zeros((len(pose), config.NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    VideoKeypoints(pose=pose, pose_world=world, ball=ball, fps=24.0, width=1280, height=720).save(path)


def _row(file, label, source):
    return {field: "" for field in dataset.MANIFEST_FIELDS} | {
        "file": file,
        "label": label,
        "source": source,
        "clip_id": 0,
    }


def test_load_impact_dataset(tmp_path, make_clip):
    pose, ball = make_clip(num_frames=20, impact_frame=10)
    _save_clip(tmp_path / "back" / "a.npz", pose, ball)
    _save_clip(tmp_path / "front" / "b.npz", pose, ball)
    _save_clip(tmp_path / "middle" / "c.npz", pose, ball)
    no_ball = np.full_like(ball, np.nan)
    _save_clip(tmp_path / "back" / "d.npz", pose, no_ball)  # 임팩트를 못 찾는 영상
    dataset.write_manifest(
        tmp_path,
        [
            _row("back/a.npz", "back", "thesis63"),
            _row("front/b.npz", "front", "thesis63"),
            _row("middle/c.npz", "middle", "extra60"),
            _row("back/d.npz", "back", "thesis63"),
        ],
    )

    x, y, names = dataset.load_impact_dataset(tmp_path)
    assert x.shape == (3, config.IMPACT_SEQ_LEN, config.IMPACT_INPUT_SIZE)
    assert sorted(names) == ["back/a.npz", "front/b.npz", "middle/c.npz"]
    # 클래스 번호는 config.IMPACT_LABELS 순서(back=0, front=1, middle=2)
    assert dict(zip(names, y.tolist(), strict=True)) == {"back/a.npz": 0, "front/b.npz": 1, "middle/c.npz": 2}

    x, y, names = dataset.load_impact_dataset(tmp_path, source="extra60")
    assert names == ["middle/c.npz"]


def test_load_linear_dataset(tmp_path, make_body_clip):
    pose, ball = make_body_clip(num_frames=20, impact_frame=10)
    _save_clip(tmp_path / "back" / "a.npz", pose, ball)
    _save_clip(tmp_path / "front" / "b.npz", pose, ball)
    _save_clip(tmp_path / "middle" / "c.npz", pose, np.full_like(ball, np.nan))  # 공이 없는 영상
    dataset.write_manifest(
        tmp_path,
        [
            _row("back/a.npz", "back", "thesis63"),
            _row("front/b.npz", "front", "extra60"),
            _row("middle/c.npz", "middle", "thesis63"),
        ],
    )

    x, y, names = dataset.load_linear_dataset("weight", tmp_path)
    assert x.shape == (2, 66)  # 관절 33개의 (x, y)
    assert dict(zip(names, y.tolist(), strict=True)) == {"back/a.npz": 0, "front/b.npz": 1}

    x, y, names = dataset.load_linear_dataset("weight", tmp_path, source="extra60")
    assert names == ["front/b.npz"]


def test_load_linear_dataset_for_foot(tmp_path, make_body_clip):
    pose, ball = make_body_clip(num_frames=20, impact_frame=10)
    _save_clip(tmp_path / "aligned" / "a.npz", pose, ball)
    _save_clip(tmp_path / "too_far" / "b.npz", pose, ball)
    dataset.write_manifest(
        tmp_path, [_row("aligned/a.npz", "aligned", "t"), _row("too_far/b.npz", "too_far", "t")]
    )

    x, y, _ = dataset.load_linear_dataset("foot", tmp_path)
    assert x.shape == (2, 6)  # 왼발 발목, 뒤꿈치, 발끝의 (x, y)
    assert sorted(y.tolist()) == [0, 4]  # aligned=0, too_far=4


def test_keypoints_save_load_roundtrip(tmp_path, make_clip):
    pose, ball = make_clip(num_frames=8, impact_frame=4)
    pose[3] = np.nan
    ball[5] = np.nan
    _save_clip(tmp_path / "clip.npz", pose, ball)

    loaded = VideoKeypoints.load(tmp_path / "clip.npz")
    assert loaded.num_frames == 8
    assert (loaded.fps, loaded.width, loaded.height) == (24.0, 1280, 720)
    assert loaded.pose_detected.tolist() == [True, True, True, False, True, True, True, True]
    assert loaded.ball_detected.tolist() == [True, True, True, True, True, False, True, True]
