#!/bin/bash
# Quantitative Investment Model - 웹 앱 실행 스크립트

cd "$(dirname "$0")"

# FSEvents 에러 회피 (macOS)
export WATCHDOG_OBSERVER_POLLING=1

# Streamlit 실행
echo "웹 앱을 시작합니다..."
echo "브라우저에서 http://localhost:8501 로 접속하세요."
echo "다른 기기에서 접속: http://$(hostname -f 2>/dev/null || echo 'localhost'):8501"
echo ""
streamlit run app.py "$@"
