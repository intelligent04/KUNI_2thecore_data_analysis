#!/bin/bash
# Ubuntu 서버 환경에서 KUNI 2thecore 데이터 분석 서버 설정 스크립트

echo "=== KUNI 2thecore 데이터 분석 서버 Ubuntu 설정 시작 ==="

# 시스템 패키지 업데이트
echo "1. 시스템 패키지 업데이트 중..."
sudo apt update
sudo apt upgrade -y

# Python 3.9+ 및 필수 패키지 설치
echo "2. Python 및 필수 패키지 설치 중..."
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y build-essential libmysqlclient-dev pkg-config

# 한글 폰트 패키지 설치 (중요!)
echo "3. 한글 폰트 설치 중..."
sudo apt install -y fonts-noto-cjk fonts-nanum fonts-liberation
sudo apt install -y fontconfig

# 폰트 캐시 업데이트
echo "4. 폰트 캐시 업데이트 중..."
sudo fc-cache -fv

# 설치된 한글 폰트 확인
echo "5. 설치된 한글 폰트 확인:"
fc-list :lang=ko | head -5

# 가상환경 생성 및 활성화 (선택사항)
echo "6. Python 가상환경 설정 (권장):"
echo "python3 -m venv venv"
echo "source venv/bin/activate"
echo "pip install -r requirements.txt"

# 환경변수 설정 안내
echo "7. 환경변수 설정 (.env 파일 생성 필요):"
echo "DB_HOST=your_mysql_host"
echo "DB_USER=your_mysql_user"
echo "DB_PASSWORD=your_mysql_password"
echo "DB_NAME=your_database_name"
echo "DB_PORT=3306"

# 서버 실행 안내
echo "8. 서버 실행 방법:"
echo "python run_server.py  # 개발 서버"
echo "# 또는 프로덕션용:"
echo "gunicorn -w 4 -b 0.0.0.0:5000 app:app"

echo "=== Ubuntu 설정 완료 ==="
echo "한글 폰트가 정상적으로 설치되었는지 확인하세요:"
echo "fc-list :lang=ko"