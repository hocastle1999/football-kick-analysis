// 화면 흐름: 영상 선택 → 업로드 → 진행률을 1초마다 물어봄(폴링) → 결과 표시
// 주소에 ?job=작업번호 가 있으면 새로고침해도 같은 결과를 다시 보여 준다.

const $ = (id) => document.getElementById(id);

const WEIGHT_NAMES = { back: "뒤쪽", middle: "가운데", front: "앞쪽" };
// 무게중심은 공을 지나는 수직선을 기준으로 몸이 어느 쪽에 실려 있는지를 뜻한다
const WEIGHT_DESCRIPTIONS = {
  back: "몸이 공보다 뒤쪽에 실린 자세입니다.",
  middle: "몸이 공 바로 위에 실린 자세입니다.",
  front: "몸이 공보다 앞쪽에 실린 자세입니다.",
};
const FOOT_NAMES = {
  aligned: "공 옆에 나란히",
  behind: "공보다 뒤",
  in_front: "공보다 앞",
  too_close: "공에 너무 가까움",
  too_far: "공에서 너무 멂",
};
const KICK_NAMES = { long: "롱 패스", short_pass: "짧은 패스", shoot: "슈팅" };
const POLL_INTERVAL_MS = 1000;

let selectedFile = null;
let impactTime = 0;

// ---------- 영상 선택 ----------
function selectFile(file) {
  if (!file) return;
  selectedFile = file;
  $("drop-title").textContent = file.name;
  $("drop-hint").textContent = `${(file.size / 1024 / 1024).toFixed(1)}MB · 다른 영상을 고르려면 다시 누르세요`;
  $("start-button").disabled = false;
  $("upload-error").hidden = true;
}

$("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
const dropZone = $("drop-zone");
["dragenter", "dragover"].forEach((name) =>
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("over");
  })
);
["dragleave", "drop"].forEach((name) =>
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("over");
  })
);
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

// ---------- 업로드 ----------
$("start-button").addEventListener("click", async () => {
  const form = new FormData();
  form.append("video", selectedFile);
  form.append("kick_type", document.querySelector("input[name=kick]:checked").value);

  show("progress-view");
  setProgress("영상을 올리는 중입니다", 0, "");
  try {
    const response = await fetch("/api/analyses", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "업로드에 실패했습니다.");
    history.replaceState(null, "", `?job=${body.id}`);
    poll(body.id);
  } catch (error) {
    showUploadError(error.message);
  }
});

function showUploadError(message) {
  show("upload-view");
  $("upload-error").textContent = message;
  $("upload-error").hidden = false;
}

// ---------- 진행률 확인 ----------
async function poll(jobId) {
  let job;
  try {
    const response = await fetch(`/api/analyses/${jobId}`);
    if (response.status === 404) {
      // 서버를 다시 켰거나 오래된 작업이면 기록이 없다
      history.replaceState(null, "", location.pathname);
      showUploadError("분석 기록을 찾을 수 없습니다. 영상을 다시 올려 주세요.");
      return;
    }
    job = await response.json();
  } catch {
    showUploadError("서버에 연결할 수 없습니다.");
    return;
  }

  if (job.status === "done") {
    showResult(job);
  } else if (job.status === "failed") {
    history.replaceState(null, "", location.pathname);
    showUploadError(job.error);
  } else {
    show("progress-view");
    const { done, total } = job.progress;
    if (job.status === "queued") {
      setProgress("앞선 분석이 끝나기를 기다리는 중입니다", 0, "");
    } else {
      const percent = total ? (done / total) * 100 : 0;
      setProgress("영상을 분석하는 중입니다", percent, total ? `${done} / ${total} 프레임` : "모델 준비 중");
    }
    setTimeout(() => poll(jobId), POLL_INTERVAL_MS);
  }
}

function setProgress(title, percent, text) {
  $("progress-title").textContent = title;
  $("bar-fill").style.width = `${percent}%`;
  $("progress-text").textContent = text;
}

// ---------- 결과 ----------
function showResult(job) {
  const result = job.result;
  impactTime = result.impact_time; // 발이 공에 닿는 시점(초). 프레임 사이일 수 있다

  setVerdict(
    "weight",
    result.weight_label,
    result.weight_confidence,
    WEIGHT_NAMES,
    result.recommended_weight,
    `${KICK_NAMES[job.kick_type]}에 알맞은 무게중심입니다.`,
    `${KICK_NAMES[job.kick_type]}에서는 무게중심을 ${WEIGHT_NAMES[result.recommended_weight]}에 두는 것이 좋습니다.`
  );
  setVerdict(
    "foot",
    result.foot_label,
    result.foot_confidence,
    FOOT_NAMES,
    "aligned",
    "디딤발 위치가 알맞습니다.",
    `디딤발을 "${FOOT_NAMES.aligned}" 두는 것이 좋습니다.`
  );

  $("weight-description").textContent = WEIGHT_DESCRIPTIONS[result.weight_label] || "";

  $("impact-image").src = `/api/analyses/${job.id}/impact.jpg`;
  $("video").src = `/api/analyses/${job.id}/video`;
  $("video-error").hidden = true;
  selectTab("image");

  const t = result.timings;
  const perFrame = (t.extract / result.frames) * 1000;
  const details = [
    ["영상", `${job.filename} · ${result.width}×${result.height} · ${result.fps}fps · ${result.frames}프레임`],
    ["킥 종류", KICK_NAMES[job.kick_type]],
    ["임팩트", `${result.impact_time.toFixed(2)}초 (${result.impact_frame}번 프레임 직후)`],
    ["전체 분석 시간", `${t.total.toFixed(1)}초`],
    ["좌표 추출", `${t.extract.toFixed(1)}초 (프레임당 ${perFrame.toFixed(0)}ms)`],
    ["임팩트 추정과 분류", `${(t.classify * 1000).toFixed(1)}ms`],
  ];
  $("details").replaceChildren(
    ...details.flatMap(([name, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = name;
      dd.textContent = value; // textContent를 써서 파일 이름에 든 문자가 HTML로 해석되지 않게 한다
      return [dt, dd];
    })
  );

  show("result-view");
}

const UNSURE_BELOW = 0.6; // 분류 확률이 이보다 낮으면 "확실하지 않음"을 함께 보여 준다

function setVerdict(prefix, label, confidence, names, ideal, goodText, warnText) {
  const card = $(`${prefix}-card`);
  card.classList.remove("good", "warn", "unknown");
  if (!label) {
    card.classList.add("unknown");
    $(`${prefix}-value`).textContent = "판단하지 못함";
    $(`${prefix}-advice`).textContent = "접촉 순간의 관절을 찾지 못했거나 모델이 준비되지 않았습니다.";
    $(`${prefix}-confidence`).textContent = "";
    return;
  }
  const good = label === ideal;
  card.classList.add(good ? "good" : "warn");
  $(`${prefix}-value`).textContent = names[label] || label;
  $(`${prefix}-advice`).textContent = good ? goodText : warnText;
  const percent = Math.round(confidence * 100);
  $(`${prefix}-confidence`).textContent =
    `분류 확률 ${percent}%` + (confidence < UNSURE_BELOW ? " · 확실하지 않음" : "");
}

// ---------- 탭과 영상 ----------
function selectTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  $("tab-image").hidden = name !== "image";
  $("tab-video").hidden = name !== "video";
  if (name !== "video") $("video").pause();
}
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.tab)));

$("jump-button").addEventListener("click", () => {
  $("video").pause();
  $("video").currentTime = impactTime;
});
$("slow-check").addEventListener("change", (event) => {
  $("video").playbackRate = event.target.checked ? 0.5 : 1;
});
$("video").addEventListener("error", () => {
  $("video-error").hidden = false;
});

$("again-button").addEventListener("click", () => {
  history.replaceState(null, "", location.pathname);
  $("video").removeAttribute("src");
  show("upload-view");
});

// ---------- 공통 ----------
function show(viewId) {
  for (const id of ["upload-view", "progress-view", "result-view"]) $(id).hidden = id !== viewId;
  $("guide-view").hidden = viewId !== "upload-view"; // 촬영 안내는 영상 선택 화면에서만 보여 준다
}

const jobFromUrl = new URLSearchParams(location.search).get("job");
if (jobFromUrl) poll(jobFromUrl);
