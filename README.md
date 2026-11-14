# 축구 킥 자세 분석

스마트폰으로 옆에서 찍은 킥 영상에서 임팩트 순간을 찾아 **무게중심**(앞 / 가운데 / 뒤)과
**디딤발 위치**(나란히 / 앞 / 뒤 / 너무 가까움 / 너무 멂)를 분류하고, 킥 종류에 맞는 자세인지 알려 준다.

3인 팀 졸업 프로젝트(2025)로 만든 코드를 다시 정리하고 검증한 것이다.
논문 당시의 결과 그래프는 `docs/thesis_results/`에 그대로 두었다.

- 무엇을 어떻게 다시 쟀고 왜 모델을 바꿨는지: [docs/evaluation.md](docs/evaluation.md)
- 찾은 문제와 해결 과정: [docs/troubleshooting.md](docs/troubleshooting.md)

## 동작 방식

```
영상 ─▶ 프레임별 좌표 추출 ─▶ 임팩트 추정 ─▶ 좌표 정규화 ─▶ 분류 ─▶ 피드백
        MediaPipe BlazePose      발끝이 공의        공이 원점,        로지스틱
        (관절 33개)              가장자리를         몸통 길이가 1      회귀
        YOLOv8n (공)             지나는 시점
```

1. **좌표 추출**: 프레임마다 관절 33개와 공의 위치를 뽑는다.
2. **임팩트 추정**: 차기 전 멈춰 있는 공의 자리를 정하고, 차는 발끝이 공의 가장자리를 지나는 시점을
   프레임 사이까지 계산한다(예: 23.4번째 프레임).
3. **정규화**: 그 시점의 관절 좌표를 공이 원점이고 몸통 길이가 1인 좌표로 바꾼다.
   사람이 화면 어디에 있든, 카메라가 얼마나 멀든 같은 자세면 같은 값이 된다.
4. **분류**: 로지스틱 회귀. 무게중심은 관절 33개의 (x, y) 66개 값, 디딤발은 왼발 위치 6개 값을 쓴다.
5. **피드백**: 킥 종류별 권장 무게중심(롱 패스는 뒤, 짧은 패스는 가운데, 슈팅은 앞)과 비교한다.

논문의 방식(화면 좌표 + LSTM + 동적 양자화)은 비교용으로 남겨 두었다(`scripts/train_lstm.py`, `kick_analysis/models.py`).

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `kick_analysis/` | 공용 모듈. 학습, 웹 서버, GUI가 모두 이 코드를 쓴다 |
| `kick_analysis/extraction.py` | 영상 → 프레임별 관절과 공 좌표 |
| `kick_analysis/impact.py` | 임팩트 추정 |
| `kick_analysis/features.py` | 좌표 정규화와 분류기 입력 생성 |
| `kick_analysis/linear.py` | 로지스틱 회귀 분류기(학습, 저장, 추론) |
| `kick_analysis/pipeline.py` | 영상 하나를 분석하는 전체 흐름과 피드백 문장 |
| `kick_analysis/models.py` | (비교용) LSTM 정의와 양자화 |
| `scripts/` | 키포인트 추출, 학습, 평가, 명령줄 분석, 관절 중요도 |
| `web/` | 웹 서버(FastAPI)와 화면 |
| `gui/` | 데스크톱 GUI(Tkinter) |
| `tests/` | pytest |

## 설치

Python 3.10, Windows 기준이다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## 사용법

모든 명령은 프로젝트 루트에서 실행한다.

### 1. 키포인트 추출

원본 영상은 저장소에 없다(사람이 찍혀 있어 올리지 않는다).
대신 영상에서 뽑은 키포인트(관절과 공의 좌표)가 `data/keypoints/`에 들어 있어서,
영상이 없어도 2번(학습과 평가)부터 바로 실행할 수 있다.

새 영상으로 키포인트를 뽑으려면 영상 폴더를 지정한다.

```powershell
# 무게중심 영상: 파일 이름이 kick_video(back_1).mp4 같은 형식
.\.venv\Scripts\python.exe -m scripts.extract_keypoints --videos "영상 폴더" --source thesis63
# 디딤발 영상: 폴더 이름이 라벨(aligned, behind, ...)
.\.venv\Scripts\python.exe -m scripts.extract_keypoints --videos "영상 폴더" --out data/keypoints/foot --source thesis50
```

결과는 `data/keypoints/<종류>/<라벨>/*.npz`와 `manifest.csv`(영상 목록)로 저장된다.

### 2. 학습과 평가

```powershell
.\.venv\Scripts\python.exe -m scripts.train --task weight
.\.venv\Scripts\python.exe -m scripts.train --task foot
.\.venv\Scripts\python.exe -m scripts.evaluate --task impact
```

`train`은 가진 영상 전부로 학습해 `models/weights/`에 저장한다.
`evaluate`는 여러 방법을 교차 검증과 사람 단위 평가로 비교한다.

### 3. 분석

웹:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.app:app --port 8000
```

브라우저에서 <http://localhost:8000>을 열고 영상을 올린다.
같은 와이파이의 휴대폰에서 쓰려면 `--host 0.0.0.0`을 붙이고 `http://<PC의 IP>:8000`으로 접속한다.

명령줄과 데스크톱 GUI:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze "영상.mp4" --kick long
.\.venv\Scripts\python.exe -m gui.app
```

### 테스트와 린트

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

## 웹 서버 구조

| 요청 | 설명 |
|---|---|
| `POST /api/analyses` | 영상과 킥 종류를 올린다. 작업 번호를 바로 돌려준다(202) |
| `GET /api/analyses/{id}` | 진행률, 끝났으면 결과 |
| `GET /api/analyses/{id}/impact.jpg` | 임팩트 순간 이미지(관절과 공 표시) |
| `GET /api/analyses/{id}/video` | 올린 영상(브라우저 재생용) |

- 분석에 수십 초가 걸려서 업로드 요청은 바로 끝내고, 분석은 작업 스레드 하나가 차례대로 처리한다.
  화면은 1초마다 진행률을 물어본다.
- 작업 상태는 메모리에 둔다(최근 20개). 올린 영상은 임시 폴더에 두고 서버가 꺼지면 지운다.
  혼자 쓰는 로컬 도구라는 전제이며, 여러 사람이 쓰는 서비스라면 저장소와 작업 큐가 필요하다.
- 업로드는 확장자(mp4, mov, avi)와 크기(200MB)를 확인하고, 저장 파일 이름은 서버가 정한다.

## 성능

자세한 표와 평가 방식은 [docs/evaluation.md](docs/evaluation.md)에 있다.

무게중심 (영상 123개, 5명):

| 방법 | 반복 교차 검증 | 사람 단위 | 63개로 학습 → 추가 60개 |
|---|---|---|---|
| 논문 방식 (화면 좌표 + LSTM) | 0.75 | 0.56 | 0.77 |
| 정규화 좌표 + LSTM | 0.94 | 0.69 | 0.83 |
| **지금 방식 (정규화 자세 + 로지스틱 회귀)** | **0.97** | **0.76** | **0.95** |

디딤발 (aligned를 뺀 4클래스, 영상 40개, 2명):

| 방법 | 반복 교차 검증 | 사람 단위 |
|---|---|---|
| 논문 방식 (화면 좌표 + LSTM) | 0.72 | 0.47 |
| **지금 방식 (왼발 위치 + 로지스틱 회귀)** | 0.84 | **0.75** |

- "사람 단위"는 한 사람의 영상을 전부 빼고 학습한 뒤 그 사람으로 시험한 값이다. 처음 보는 사람에 대한 성능에 가깝다.
- 디딤발의 aligned 영상은 다른 날 찍은 것이라, 5클래스 정확도는 촬영 조건을 구분한 결과가 섞여 있다.
  그래서 디딤발 판정은 화면에 참고용으로 표시한다.

속도 (노트북 CPU, GPU 없음):

| 영상 | 처음 | 지금 |
|---|---|---|
| 1280×720, 64프레임 (2.7초) | 19.0초 | 5.4초 (3.5배) |
| 1920×1080, 133프레임 (4.4초) | 28.4초 | 7.7초 (3.7배) |

분석 시간은 거의 전부 좌표 추출에 쓰인다. 분류 모델을 줄이는 것(양자화 등)은 전체 시간에 영향이 없다.
그래서 좌표 추출을 줄였다: 공은 자리를 확인하면 검출을 멈추고, 포즈는 빠른 모델로 전체를 훑은 뒤
임팩트 주변만 정밀한 모델로 다시 뽑는다. 정확도는 그대로다. 과정은 [docs/performance.md](docs/performance.md)에 있다.
모델 파일은 무게중심 4KB, 디딤발 2KB다(LSTM은 172KB, 384KB였다).

## 알려진 한계

- 오른발 킥, 옆에서 고정 촬영한 영상만 다룬다.
- 학습에 쓴 사람이 5명(디딤발은 4명)뿐이다. 처음 보는 사람에 대한 정확도는 0.75 안팎이다.
- 추가 영상에서는 킥 종류와 무게중심 라벨이 항상 같이 바뀐다. 모델이 무게중심이 아니라
  킥 종류의 다른 특징을 보고 맞혔을 가능성을 배제하지 못했다.
- 디딤발의 aligned 클래스는 같은 조건으로 다시 찍어야 믿을 수 있다.
- 화면에 나오는 분류 확률은 실제 맞을 가능성보다 높게 나온다.
- 실시간이 아니다. 영상 길이의 약 2배가 걸린다.

## 다음 단계

1. 데이터 보강: 같은 킥 종류 안에서 무게중심만 바꾼 영상, 같은 조건의 aligned 영상, 더 많은 사람
2. 왼발 킥 지원
3. 좌표 추출을 더 줄이기: 빠른 모델을 2프레임마다 돌리기, 영상을 줄여서 읽기
