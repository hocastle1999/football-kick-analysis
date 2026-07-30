# ⚽ Football Kick Posture Analysis

포즈 모델과 딥러닝(LSTM)을 활용하여 축구 킥 동작 시 무게중심과 디딤발 위치를 실시간으로 분석하고 교정 피드백을 제공하는 시스템입니다[cite: 2].

## 🛠️ Tech Stack
* Python[cite: 2]
* PyTorch[cite: 2]
* MediaPipe BlazePose & YOLOv8n[cite: 2]
* Dynamic Quantization[cite: 2]

## ✨ Features
* 스마트폰 영상을 활용한 신체 주요 관절 및 공의 3D 좌표 추출[cite: 2]
* 임팩트 전후 무게중심(ImpactLSTM) 및 디딤발 위치(FootLSTM) 정밀 분석[cite: 2]
* 동적 양자화를 통한 모델 경량화 및 CPU 환경 실시간 추론 최적화[cite: 2]
