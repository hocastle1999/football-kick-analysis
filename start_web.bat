@echo off
rem 축구 킥 자세 분석 웹 서버를 켜고 브라우저를 연다. 끌 때는 이 창에서 Ctrl+C.
chcp 65001 >nul
title 축구 킥 자세 분석 (이 창을 닫으면 서버가 꺼집니다)

rem 어디서 실행하든 이 파일이 있는 프로젝트 폴더로 옮긴다
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo 가상환경이 없습니다. README의 "설치"를 먼저 실행하세요.
    pause
    exit /b 1
)
if not exist "models\weights\weight_linear.npz" (
    echo 학습된 분류기가 없어 먼저 학습합니다...
    ".venv\Scripts\python.exe" -m scripts.train --task weight
    ".venv\Scripts\python.exe" -m scripts.train --task foot
)

rem 서버가 이미 켜져 있으면 다시 켜지 않고 브라우저만 연다(같은 포트로 두 번 켜면 오류가 난다)
curl -s -o nul http://localhost:8000/ && (
    echo 서버가 이미 켜져 있습니다. 브라우저를 엽니다.
    start http://localhost:8000
    exit /b 0
)

echo 서버를 켜는 중입니다. 잠시 뒤 브라우저가 열립니다.
rem 서버가 뜰 시간을 준 뒤 브라우저를 연다(서버는 아래 줄에서 이 창을 차지한다)
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8000"

".venv\Scripts\python.exe" -m uvicorn web.app:app --port 8000
pause
