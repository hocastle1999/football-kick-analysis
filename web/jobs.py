"""분석 작업 관리.

영상 분석은 수십 초가 걸려서, 업로드 요청이 끝날 때까지 기다리게 하면 브라우저가 멈춘 것처럼 보인다.
그래서 요청은 작업 번호만 바로 돌려주고, 실제 분석은 뒤에서 돌린다. 브라우저는 작업 번호로 진행률을 물어본다.

작업 상태는 메모리(dict)에 둔다. 혼자 쓰는 로컬 도구라서 서버를 끄면 사라져도 문제가 없기 때문이다.
여러 서버가 같이 쓰거나 재시작 뒤에도 남아야 한다면 DB나 Redis로 옮겨야 한다.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from kick_analysis.pipeline import AnalysisError, AnalysisResult, make_feedback

MAX_JOBS = 20  # 이보다 많아지면 가장 오래된 작업과 그 영상을 지운다


@dataclass
class Job:
    id: str
    kick_type: str
    filename: str  # 사용자가 올린 원래 파일 이름(화면 표시용)
    video_path: Path  # 서버에 저장한 경로(이름은 서버가 정한다)
    status: str = "queued"  # queued → running → done / failed
    progress_done: int = 0
    progress_total: int = 0
    error: str | None = None
    result: AnalysisResult | None = None
    feedback: list[str] = field(default_factory=list)


class JobManager:
    """작업을 등록하고, 작업 스레드 하나로 차례대로 분석한다.

    스레드를 하나만 쓰는 이유: 분석이 CPU를 거의 다 쓰므로 동시에 돌려도 빨라지지 않고,
    포즈 모델과 YOLO 객체를 여러 스레드가 함께 쓰는 것도 안전하지 않다.
    """

    def __init__(self, analyzer, upload_dir: Path, max_jobs: int = MAX_JOBS) -> None:
        self._analyzer = analyzer
        self._upload_dir = upload_dir
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()  # 요청 스레드와 작업 스레드가 같은 dict를 건드리므로 잠근다
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")

    def new_job_dir(self) -> tuple[str, Path]:
        """새 작업 번호와 그 작업의 영상을 둘 폴더를 만든다."""
        job_id = uuid.uuid4().hex
        job_dir = self._upload_dir / job_id
        job_dir.mkdir(parents=True)
        return job_id, job_dir

    def submit(self, job_id: str, kick_type: str, filename: str, video_path: Path) -> Job:
        job = Job(id=job_id, kick_type=kick_type, filename=filename, video_path=video_path)
        with self._lock:
            self._jobs[job_id] = job
            while len(self._jobs) > self._max_jobs:
                _, old = self._jobs.popitem(last=False)
                shutil.rmtree(old.video_path.parent, ignore_errors=True)
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job) -> None:
        job.status = "running"

        def on_progress(done: int, total: int) -> None:
            job.progress_done, job.progress_total = done, total

        try:
            result = self._analyzer.analyze(job.video_path, on_progress=on_progress)
        except (AnalysisError, OSError, ValueError) as error:
            job.error = str(error)
            job.status = "failed"
            return
        except Exception:  # 예상하지 못한 오류도 작업 실패로 남겨 브라우저가 계속 기다리지 않게 한다
            job.error = "분석 중 알 수 없는 오류가 발생했습니다."
            job.status = "failed"
            raise

        job.result = result
        job.feedback = make_feedback(job.kick_type, result.weight_label, result.foot_label)
        job.status = "done"  # 결과를 다 채운 뒤 마지막에 상태를 바꾼다
