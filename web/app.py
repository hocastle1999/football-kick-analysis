"""축구 킥 분석 웹 서버 (FastAPI).

실행 (프로젝트 루트에서):
    python -m uvicorn web.app:app --port 8000
그다음 브라우저에서 http://localhost:8000 을 연다.

흐름:
    POST /api/analyses                  영상 업로드 → 작업 번호를 바로 돌려준다(202)
    GET  /api/analyses/{id}             진행률, 끝났으면 결과
    GET  /api/analyses/{id}/impact.jpg  임팩트 순간 이미지(관절과 공 표시)
    GET  /api/analyses/{id}/video       올린 영상(브라우저 재생용)
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from kick_analysis.pipeline import RECOMMENDED_WEIGHT
from web.jobs import Job, JobManager
from web.render import render_frame

STATIC_DIR = Path(__file__).parent / "static"
WEIGHTS_DIR = Path("models/weights")
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB
CHUNK_SIZE = 1024 * 1024


def create_app(analyzer=None) -> FastAPI:
    """앱을 만든다. analyzer를 주지 않으면 서버가 시작될 때 실제 모델을 불러온다.

    테스트에서는 가짜 analyzer를 넣어 모델 없이 API만 검사한다.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 서버가 뜰 때 한 번만 모델을 불러온다(요청마다 불러오면 매번 몇 초씩 걸린다)
        loaded = analyzer
        if loaded is None:
            from kick_analysis.pipeline import KickAnalyzer

            loaded = KickAnalyzer(WEIGHTS_DIR / "weight_linear.npz", WEIGHTS_DIR / "foot_linear.npz")
        upload_dir = Path(tempfile.mkdtemp(prefix="kick-analysis-"))
        app.state.jobs = JobManager(loaded, upload_dir)
        yield
        # 서버가 꺼질 때 올라온 영상을 모두 지운다
        app.state.jobs.shutdown()
        shutil.rmtree(upload_dir, ignore_errors=True)

    app = FastAPI(title="축구 킥 분석", lifespan=lifespan)

    @app.post("/api/analyses", status_code=202)
    async def create_analysis(request: Request, video: UploadFile = File(...), kick_type: str = Form(...)):
        if kick_type not in RECOMMENDED_WEIGHT:
            raise HTTPException(422, f"kick_type은 {sorted(RECOMMENDED_WEIGHT)} 중 하나여야 합니다.")
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, "mp4, mov, avi 영상만 올릴 수 있습니다.")

        jobs: JobManager = request.app.state.jobs
        job_id, job_dir = jobs.new_job_dir()
        # 저장 이름은 서버가 정한다. 사용자가 보낸 파일 이름을 경로에 쓰면
        # "../" 같은 값으로 다른 위치에 파일을 쓰게 만들 수 있다
        video_path = job_dir / f"video{suffix}"

        # 한꺼번에 메모리에 올리지 않고 1MB씩 받아 쓰면서 크기를 확인한다
        size = 0
        with open(video_path, "wb") as out:
            while chunk := await video.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(413, "영상이 너무 큽니다(최대 200MB).")
                out.write(chunk)
        if size == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(422, "빈 파일입니다.")

        job = jobs.submit(job_id, kick_type, video.filename or "video", video_path)
        return {"id": job.id, "status": job.status}

    @app.get("/api/analyses/{job_id}")
    def get_analysis(job_id: str, request: Request):
        return _to_response(_find_job(request, job_id))

    @app.get("/api/analyses/{job_id}/impact.jpg")
    def get_impact_image(job_id: str, request: Request):
        job = _find_job(request, job_id)
        if job.status != "done":
            raise HTTPException(409, "아직 분석이 끝나지 않았습니다.")
        image = render_frame(job.video_path, job.result.keypoints, job.result.impact_frame)
        return Response(image, media_type="image/jpeg")

    @app.get("/api/analyses/{job_id}/video")
    def get_video(job_id: str, request: Request):
        job = _find_job(request, job_id)
        # FileResponse는 Range 요청을 처리하므로 브라우저에서 원하는 위치로 바로 이동할 수 있다
        return FileResponse(job.video_path)

    # API 경로를 먼저 등록한 뒤, 나머지 경로는 정적 파일(index.html 등)로 처리한다
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _find_job(request: Request, job_id: str) -> Job:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "분석 작업을 찾을 수 없습니다.")
    return job


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _to_response(job: Job) -> dict:
    body = {
        "id": job.id,
        "status": job.status,
        "filename": job.filename,
        "kick_type": job.kick_type,
        "progress": {"done": job.progress_done, "total": job.progress_total},
        "error": job.error,
        "result": None,
    }
    if job.status == "done":
        result = job.result
        keypoints = result.keypoints
        body["result"] = {
            "impact_frame": result.impact_frame,
            "impact_time": round(result.impact_time / keypoints.fps, 3),  # 초 단위
            "weight_label": result.weight_label,
            "weight_confidence": _round(result.weight_confidence),
            "recommended_weight": RECOMMENDED_WEIGHT[job.kick_type],
            "foot_label": result.foot_label,
            "foot_confidence": _round(result.foot_confidence),
            "feedback": job.feedback,
            "frames": keypoints.num_frames,
            "fps": round(keypoints.fps, 2),
            "width": keypoints.width,
            "height": keypoints.height,
            "timings": {name: round(seconds, 4) for name, seconds in result.timings.items()},
        }
    return body


app = create_app()
