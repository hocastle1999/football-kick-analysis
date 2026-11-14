"""웹 API 테스트. 실제 모델 대신 가짜 분석기를 넣어 요청과 응답의 흐름만 검사한다."""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kick_analysis import config
from kick_analysis.extraction import VideoKeypoints
from kick_analysis.pipeline import AnalysisError, AnalysisResult
from web.app import create_app


class FakeAnalyzer:
    """정해진 결과를 돌려주는 분석기. fail=True면 분석 실패를 흉내 낸다."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def analyze(self, video_path, on_progress=None):
        if self.fail:
            raise AnalysisError("발과 공이 함께 잡힌 프레임이 없어 임팩트를 찾지 못했습니다.")
        frames = 10
        if on_progress:
            on_progress(frames, frames)
        pose = np.full((frames, config.NUM_POSE_LANDMARKS, 4), 0.5, dtype=np.float32)
        world = np.zeros((frames, config.NUM_POSE_LANDMARKS, 3), dtype=np.float32)
        ball = np.tile(np.array([0.5, 0.8, 0.1, 0.1, 0.9], dtype=np.float32), (frames, 1))
        keypoints = VideoKeypoints(pose, world, ball, fps=24.0, width=64, height=48)
        return AnalysisResult(
            impact_frame=4,
            impact_time=3.6,
            weight_label="middle",
            weight_confidence=0.8234,
            foot_label=None,
            foot_confidence=None,
            keypoints=keypoints,
            timings={"extract": 1.5, "classify": 0.002, "total": 1.6},
        )


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def client():
    # with 문 안에서 서버 시작과 종료 처리(lifespan)가 실행된다
    with TestClient(create_app(FakeAnalyzer())) as test_client:
        yield test_client


def wait_until_finished(client, job_id):
    for _ in range(100):
        body = client.get(f"/api/analyses/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("분석이 끝나지 않았습니다")


def upload(client, content, name="kick.mp4", kick_type="long"):
    return client.post(
        "/api/analyses", files={"video": (name, content, "video/mp4")}, data={"kick_type": kick_type}
    )


def test_upload_then_get_result(client, video_bytes):
    response = upload(client, video_bytes)
    assert response.status_code == 202
    job_id = response.json()["id"]

    body = wait_until_finished(client, job_id)
    assert body["status"] == "done"
    assert body["filename"] == "kick.mp4"
    assert body["progress"] == {"done": 10, "total": 10}
    result = body["result"]
    assert result["impact_frame"] == 4
    assert result["impact_time"] == pytest.approx(3.6 / 24, abs=0.001)  # 프레임 단위 3.6 → 초
    assert result["weight_label"] == "middle"
    assert result["weight_confidence"] == 0.823
    assert result["foot_confidence"] is None
    assert result["recommended_weight"] == "back"  # 롱 패스의 권장 위치
    assert result["foot_label"] is None
    assert result["feedback"][0] == "무게중심: back 권장 (현재: middle)"


def test_impact_image_and_video(client, video_bytes):
    job_id = upload(client, video_bytes).json()["id"]
    wait_until_finished(client, job_id)

    image = client.get(f"/api/analyses/{job_id}/impact.jpg")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content[:2] == b"\xff\xd8"  # JPEG 파일의 시작 바이트

    video = client.get(f"/api/analyses/{job_id}/video")
    assert video.status_code == 200
    assert video.content == video_bytes

    # 브라우저가 영상의 일부만 요청(Range)해도 처리한다
    part = client.get(f"/api/analyses/{job_id}/video", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert len(part.content) == 100


def test_failed_analysis_reports_reason(video_bytes):
    with TestClient(create_app(FakeAnalyzer(fail=True))) as client:
        job_id = upload(client, video_bytes).json()["id"]
        body = wait_until_finished(client, job_id)
        assert body["status"] == "failed"
        assert "임팩트" in body["error"]
        assert body["result"] is None
        assert client.get(f"/api/analyses/{job_id}/impact.jpg").status_code == 409


@pytest.mark.parametrize(
    ("name", "kick_type", "content", "status"),
    [
        ("kick.txt", "long", b"hello", 415),  # 영상이 아닌 확장자
        ("kick.mp4", "volley", b"data", 422),  # 없는 킥 종류
        ("kick.mp4", "long", b"", 422),  # 빈 파일
    ],
)
def test_rejects_bad_upload(client, name, kick_type, content, status):
    assert upload(client, content, name=name, kick_type=kick_type).status_code == status


def test_upload_filename_is_not_used_as_path(client, video_bytes):
    """파일 이름에 경로를 넣어도 서버가 정한 위치에만 저장한다."""
    job_id = upload(client, video_bytes, name="../../evil.mp4").json()["id"]
    wait_until_finished(client, job_id)
    job = client.app.state.jobs.get(job_id)
    assert job.video_path.name == "video.mp4"
    assert job.video_path.parent.name == job_id


def test_unknown_job_is_404(client):
    assert client.get("/api/analyses/nope").status_code == 404
    assert client.get("/api/analyses/nope/impact.jpg").status_code == 404
    assert client.get("/api/analyses/nope/video").status_code == 404


def test_serves_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "축구 킥 자세 분석" in response.text


def test_old_jobs_are_removed(video_bytes):
    with TestClient(create_app(FakeAnalyzer())) as client:
        client.app.state.jobs._max_jobs = 2
        ids = [upload(client, video_bytes).json()["id"] for _ in range(3)]
        for job_id in ids[1:]:
            wait_until_finished(client, job_id)
        assert client.get(f"/api/analyses/{ids[0]}").status_code == 404
        assert not (client.app.state.jobs._upload_dir / ids[0]).exists()
