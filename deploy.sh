#!/bin/bash

# KUNI Analysis 배포 스크립트
# 사용법: ./deploy.sh

set -e

echo "🚀 KUNI Analysis 배포 시작..."

# 변수 설정
PROJECT_DIR="/home/ubuntu/kuni-analysis"
SERVICE_NAME="kuni-analysis"

# Git pull (코드 업데이트)
echo "📥 코드 업데이트..."
cd $PROJECT_DIR
git pull origin main

# 가상환경 활성화 및 의존성 설치
echo "📦 의존성 설치..."
source venv/bin/activate
pip install -r requirements.txt

# 데이터베이스 연결 테스트
echo "🔍 데이터베이스 연결 테스트..."
python verify_setup.py

# 서비스 재시작
echo "🔄 서비스 재시작..."
sudo systemctl restart $SERVICE_NAME
sudo systemctl restart nginx

# 상태 확인
echo "✅ 배포 상태 확인..."
sudo systemctl status $SERVICE_NAME --no-pager
curl -f http://localhost/api/health || echo "❌ Health check 실패"

echo "✨ 배포 완료!"
echo "🌐 애플리케이션 접속: http://your-domain-or-ip"
echo "📚 API 문서: http://your-domain-or-ip/apidocs/"