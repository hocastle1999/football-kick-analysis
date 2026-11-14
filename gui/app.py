"""축구 킥 분석 GUI (Tkinter).

실행 (프로젝트 루트에서):
    python -m gui.app

분석은 kick_analysis.pipeline이 하고, 이 파일은 화면만 담당한다.
영상 분석은 몇 초가 걸리므로 별도 스레드에서 돌리고, 결과는 큐로 받아 화면에 반영한다.
(Tkinter 위젯은 메인 스레드에서만 건드려야 하기 때문이다.)
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

from kick_analysis.pipeline import AnalysisError, AnalysisResult, KickAnalyzer, make_feedback

WEIGHTS_DIR = Path("models/weights")
KICK_TYPES = {"롱 패스": "long", "짧은 패스": "short_pass", "슈팅": "shoot"}
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 영상 위에 한글을 쓰기 위한 글꼴
LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_FOOT = 11, 12, 32


def memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


class KickAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("축구 킥 분석기")
        self.root.geometry("750x750")
        ttk.Style(root).theme_use("clam")

        self.messages: queue.Queue = queue.Queue()  # 분석 스레드 → 화면
        self.video_path: Path | None = None
        self.analyzer: KickAnalyzer | None = None

        self._build_widgets()
        self.root.after(100, self._poll_messages)
        # 모델을 불러오는 동안 창이 멈추지 않도록 스레드에서 불러온다
        threading.Thread(target=self._load_models, daemon=True).start()

    # ---------- 화면 구성 ----------
    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        ttk.Button(top, text="비디오 선택", command=self._select_video).pack(side=tk.LEFT, padx=5)
        self.path_label = ttk.Label(top, text="선택된 파일 없음", width=50, anchor="w")
        self.path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.start_button = ttk.Button(top, text="분석 시작", command=self._start, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)

        kick_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        kick_frame.pack(fill=tk.X)
        ttk.Label(kick_frame, text="킥 종류:").pack(side=tk.LEFT, padx=(0, 10))
        self.kick_var = tk.StringVar(value=next(iter(KICK_TYPES)))
        for name in KICK_TYPES:
            ttk.Radiobutton(kick_frame, text=name, variable=self.kick_var, value=name).pack(
                side=tk.LEFT, padx=5
            )

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10)
        self.log_area = scrolledtext.ScrolledText(notebook, wrap=tk.WORD, font=("Malgun Gothic", 9))
        self.result_area = scrolledtext.ScrolledText(
            notebook, wrap=tk.WORD, font=("Malgun Gothic", 10), state=tk.DISABLED
        )
        notebook.add(self.log_area, text="로그")
        notebook.add(self.result_area, text="결과")
        self.notebook = notebook

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=10)

    # ---------- 분석 스레드에서 실행되는 부분 ----------
    def _load_models(self) -> None:
        before = memory_mb()
        try:
            self.analyzer = KickAnalyzer(WEIGHTS_DIR / "weight_linear.npz", WEIGHTS_DIR / "foot_linear.npz")
        except Exception as error:  # 모델 파일 없음, 라이브러리 오류 등
            self.messages.put(("error", f"모델을 불러오지 못했습니다: {error}"))
            return
        loaded = [
            name
            for name, model in (
                ("무게중심", self.analyzer.weight_model),
                ("디딤발", self.analyzer.foot_model),
            )
            if model is not None
        ]
        self.messages.put(("log", f"모델 준비 완료: {', '.join(loaded)}"))
        self.messages.put(("log", f"메모리: {before:.0f} MB → {memory_mb():.0f} MB"))
        self.messages.put(("ready", None))

    def _analyze(self, video_path: Path, kick_name: str) -> None:
        before = memory_mb()
        try:
            result = self.analyzer.analyze(
                video_path, on_progress=lambda done, total: self.messages.put(("progress", (done, total)))
            )
        except (AnalysisError, OSError, ValueError) as error:
            self.messages.put(("error", str(error)))
            self.messages.put(("done", None))
            return

        feedback = make_feedback(KICK_TYPES[kick_name], result.weight_label, result.foot_label)
        frames = result.keypoints.num_frames
        t = result.timings

        def show(label, confidence):
            return "판단 불가" if label is None else f"{label} (확률 {confidence:.0%})"

        lines = [
            f"--- 분석 결과 ({kick_name}) ---",
            f"비디오: {video_path.name}",
            f"임팩트: {result.impact_time:.2f}번째 프레임",
            f"무게중심: {show(result.weight_label, result.weight_confidence)}",
            f"디딤발: {show(result.foot_label, result.foot_confidence)}",
            "",
            "--- 피드백 ---",
            *feedback,
            "",
            "--- 처리 시간 ---",
            f"총 분석: {t['total']:.2f}s",
            f"  - 좌표 추출(YOLO+Pose): {t['extract']:.2f}s (프레임당 {t['extract'] / frames * 1000:.0f}ms)",
            f"  - 임팩트 추정과 분류: {t['classify'] * 1000:.2f}ms",
            "",
            "--- 메모리 사용량 (프로세스 전체) ---",
            f"분석 전: {before:.0f} MB / 분석 후: {memory_mb():.0f} MB",
        ]
        self.messages.put(("result", "\n".join(lines)))
        self.messages.put(("done", (video_path, feedback, result)))

    # ---------- 메인 스레드에서 실행되는 부분 ----------
    def _poll_messages(self) -> None:
        """큐에 쌓인 메시지를 꺼내 화면에 반영한다. 0.1초마다 다시 호출된다."""
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _handle(self, kind: str, payload) -> None:
        if kind == "log":
            self._log(payload)
        elif kind == "error":
            self._log(f"오류: {payload}")
        elif kind == "ready":
            self._update_start_button()
        elif kind == "progress":
            done, total = payload
            self.progress.config(maximum=max(total, 1), value=done)
        elif kind == "result":
            self.result_area.config(state=tk.NORMAL)
            self.result_area.delete("1.0", tk.END)
            self.result_area.insert(tk.END, payload)
            self.result_area.config(state=tk.DISABLED)
            self.notebook.select(self.result_area)
        elif kind == "done":
            self.progress.config(value=0)
            self.start_button.config(text="분석 시작")
            self._update_start_button()
            if payload and messagebox.askyesno("재생 확인", "피드백을 넣은 영상을 재생하시겠습니까?"):
                slow = messagebox.askyesno("속도 선택", "0.5배속으로 재생하시겠습니까?")
                play_with_feedback(*payload, slow_motion=slow)

    def _log(self, text: str) -> None:
        self.log_area.insert(tk.END, text + "\n")
        self.log_area.see(tk.END)

    def _update_start_button(self) -> None:
        ready = self.analyzer is not None and self.video_path is not None
        self.start_button.config(state=tk.NORMAL if ready else tk.DISABLED)

    def _select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="비디오 파일 선택",
            filetypes=(("영상", "*.mp4 *.mov *.avi"), ("모든 파일", "*.*")),
        )
        if path:
            self.video_path = Path(path)
            self.path_label.config(text=self.video_path.name)
            self._log(f"선택된 비디오: {path}")
            self._update_start_button()

    def _start(self) -> None:
        kick_name = self.kick_var.get()
        self._log(f"분석 시작: {self.video_path.name} (킥: {kick_name})")
        self.start_button.config(text="분석 중...", state=tk.DISABLED)
        threading.Thread(target=self._analyze, args=(self.video_path, kick_name), daemon=True).start()


def play_with_feedback(
    video_path: Path, feedback: list[str], result: AnalysisResult, slow_motion: bool = False
) -> None:
    """피드백 문장을 영상 위에 그려 재생한다. 임팩트 프레임에서 멈추고, q를 누르면 끝난다."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        messagebox.showerror("오류", f"비디오를 열 수 없습니다: {video_path}")
        return

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        font = None  # 글꼴이 없으면 한글 피드백은 생략한다

    keypoints = result.keypoints
    width, height = keypoints.width, keypoints.height
    pose = keypoints.pose[result.impact_frame]
    ball = keypoints.ball[result.impact_frame]

    # 무게중심 피드백은 어깨 위쪽에, 디딤발 피드백은 공(없으면 오른발) 아래쪽에 표시한다
    weight_pos = foot_pos = None
    if not np.isnan(pose[LEFT_SHOULDER, 0]):
        shoulder = (pose[LEFT_SHOULDER, :2] + pose[RIGHT_SHOULDER, :2]) / 2
        weight_pos = (int(shoulder[0] * width), int((shoulder[1] - 0.1) * height))
    anchor = ball if not np.isnan(ball[0]) else pose[RIGHT_FOOT]
    if not np.isnan(anchor[0]):
        foot_pos = (int(anchor[0] * width), int(anchor[1] * height) + 30)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps * (2 if slow_motion else 1))

    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if font:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image, "RGBA")
            for position, text, color in (
                (weight_pos, feedback[0], (255, 255, 0)),
                (foot_pos, feedback[1], (0, 255, 255)),
            ):
                if position:
                    box = draw.textbbox(position, text, font=font)
                    draw.rectangle((box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5), fill=(0, 0, 0, 128))
                    draw.text(position, text, font=font, fill=color)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if abs(frame_number - result.impact_frame) <= 2:
            cv2.putText(frame, "IMPACT", (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        is_impact = frame_number == result.impact_frame
        if is_impact:
            cv2.putText(
                frame,
                "PAUSED - press any key",
                (width // 2 - 170, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        cv2.imshow("Feedback Video", frame)
        key = cv2.waitKey(0 if is_impact else delay) & 0xFF  # 0이면 키를 누를 때까지 기다린다
        if key == ord("q"):
            break
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    root = tk.Tk()
    KickAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
