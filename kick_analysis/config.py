"""파이프라인 전체가 공유하는 상수.

학습과 추론이 서로 다른 값을 쓰면 모델 입력이 어긋나므로, 숫자는 여기 한 곳에만 둔다.
"""

# --- 포즈와 공 ---
NUM_POSE_LANDMARKS = 33  # MediaPipe BlazePose 관절 수
RIGHT_FOOT_INDEX = 32  # BlazePose 오른발 끝(right foot index). 오른발 킥만 지원한다
BALL_CLASS_ID = 32  # COCO 데이터셋의 sports ball 클래스 번호
POSE_MODEL_COMPLEXITY = 2  # 정밀한 BlazePose 모델 크기(0, 1, 2). 논문과 같은 2를 쓴다
POSE_FAST_COMPLEXITY = 0  # 영상 전체를 훑어 임팩트 위치를 찾는 빠른 모델
# 빠른 모델이 찾은 임팩트의 앞뒤 몇 프레임을 정밀한 모델로 다시 뽑을지.
# 빠른 모델의 임팩트가 정밀한 모델과 11프레임까지 차이 난 영상이 있어 앞쪽을 넉넉히 잡는다
REFINE_BEFORE = 12
REFINE_AFTER = 6
YOLO_WEIGHTS = "yolov8n.pt"
# 공이 같은 자리에서 이만큼 검출되면 남은 프레임의 공 검출을 건너뛴다(docs/performance.md)
BALL_STOP_AFTER = 10

# --- 무게중심 모델(ImpactLSTM) ---
# 임팩트 앞 2프레임 + 임팩트 + 뒤 3프레임 = 6프레임
IMPACT_SEQ_BEFORE = 2
IMPACT_SEQ_LEN = 6
IMPACT_INPUT_SIZE = NUM_POSE_LANDMARKS * 3  # 99
# 클래스 번호는 이 순서(알파벳순)로 붙인다. 순서를 바꾸면 저장된 모델과 맞지 않게 된다
IMPACT_LABELS = ("back", "front", "middle")

# --- 디딤발 모델(FootLSTM) ---
# 임팩트 앞 5프레임 + 임팩트 + 뒤 5프레임 = 11프레임
FOOT_SEQ_BEFORE = 5
FOOT_SEQ_LEN = 11
# 논문 구조를 그대로 유지한다: 관절 33개(0~32) + 빈 자리(33~99) + 공(100)
FOOT_NUM_LANDMARKS = 101
FOOT_BALL_INDEX = 100
FOOT_EXTRA_FEATURES = 2  # 임팩트 때의 발-공 거리, y 차이
FOOT_INPUT_SIZE = FOOT_NUM_LANDMARKS * 3 + FOOT_EXTRA_FEATURES  # 305
FOOT_LABELS = ("aligned", "behind", "in_front", "too_close", "too_far")

# --- 학습 기본값(논문과 동일) ---
HIDDEN_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SIZE = 0.4
SEED = 42
