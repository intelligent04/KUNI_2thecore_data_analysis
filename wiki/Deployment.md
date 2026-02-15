# 배포 가이드

> KUNI 2thecore 데이터 분석 시스템 배포 및 운영 가이드

## 배포 환경

### 지원 플랫폼
- **Windows**: Windows 10/11, Windows Server 2019+
- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 10+
- **macOS**: macOS 11 Big Sur+
- **Docker**: Docker 20.10+

### 시스템 요구사항
| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| CPU | 2 코어 | 4 코어 이상 |
| RAM | 4GB | 8GB 이상 |
| 디스크 | 10GB | 50GB SSD |
| Python | 3.9+ | 3.10+ |
| MySQL | 5.7+ | 8.0+ |

---

## Ubuntu 서버 배포

### 1. 시스템 설정

**스크립트**: [ubuntu_setup.sh](../ubuntu_setup.sh)

```bash
#!/bin/bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 설치
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y build-essential libmysqlclient-dev pkg-config

# 한글 폰트 설치 (차트 생성에 필수)
sudo apt install -y fonts-noto-cjk fonts-nanum fonts-liberation fontconfig

# 폰트 캐시 업데이트
sudo fc-cache -fv

# 설치 확인
fc-list :lang=ko | head -5
```

### 2. 프로젝트 설정

```bash
# 프로젝트 클론
cd /home/ubuntu
git clone https://github.com/your-org/KUNI_2thecore_data_analysis.git
cd KUNI_2thecore_data_analysis

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 검증
python verify_setup.py
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
DB_HOST=your_mysql_host
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=your_database_name
DB_PORT=3306
EOF

# 권한 설정
chmod 600 .env
```

### 4. Gunicorn 설정

```bash
# Gunicorn 설치
pip install gunicorn

# 테스트 실행
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 백그라운드 실행
gunicorn -w 4 -b 0.0.0.0:5000 app:app --daemon
```

### 5. Systemd 서비스 설정

`/etc/systemd/system/kuni-analysis.service`:

```ini
[Unit]
Description=KUNI 2thecore Data Analysis API
After=network.target mysql.service

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/KUNI_2thecore_data_analysis
Environment="PATH=/home/ubuntu/KUNI_2thecore_data_analysis/venv/bin"
ExecStart=/home/ubuntu/KUNI_2thecore_data_analysis/venv/bin/gunicorn \
    --workers 4 \
    --bind 0.0.0.0:5000 \
    --timeout 120 \
    --access-logfile /var/log/kuni-analysis/access.log \
    --error-logfile /var/log/kuni-analysis/error.log \
    app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 로그 디렉토리 생성
sudo mkdir -p /var/log/kuni-analysis
sudo chown ubuntu:www-data /var/log/kuni-analysis

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable kuni-analysis
sudo systemctl start kuni-analysis
sudo systemctl status kuni-analysis
```

### 6. Nginx 설정

`/etc/nginx/sites-available/kuni-analysis`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }

    # API 타임아웃 설정
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_read_timeout 300s;  # 분석 API는 더 긴 타임아웃
    }

    # 정적 파일 캐싱
    location /static/ {
        alias /home/ubuntu/KUNI_2thecore_data_analysis/static/;
        expires 30d;
    }
}
```

```bash
# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/kuni-analysis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Windows 서버 배포

### 1. 사전 요구사항

```powershell
# Python 설치 확인
python --version

# pip 업그레이드
python -m pip install --upgrade pip
```

### 2. 서비스 설정 (NSSM 사용)

```powershell
# NSSM 다운로드
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile nssm.zip
Expand-Archive nssm.zip -DestinationPath .

# 서비스 설치
nssm.exe install KuniAnalysis "C:\KUNI_2thecore_data_analysis\.venv\Scripts\python.exe"
nssm.exe set KuniAnalysis AppParameters "run_server.py"
nssm.exe set KuniAnalysis AppDirectory "C:\KUNI_2thecore_data_analysis"

# 서비스 시작
nssm.exe start KuniAnalysis
```

---

## Docker 배포

### Dockerfile

```dockerfile
FROM python:3.10-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-nanum \
    libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 폰트 캐시 업데이트
RUN fc-cache -fv

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 애플리케이션 복사
COPY . .

# 캐시 디렉토리 생성
RUN mkdir -p /app/cache

# 환경 변수
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 5000

# 실행
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DB_HOST=mysql
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - DB_NAME=${DB_NAME}
      - DB_PORT=3306
    volumes:
      - ./cache:/app/cache
    depends_on:
      - mysql
    restart: always

  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
      - MYSQL_DATABASE=${DB_NAME}
      - MYSQL_USER=${DB_USER}
      - MYSQL_PASSWORD=${DB_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
    ports:
      - "3306:3306"
    restart: always

volumes:
  mysql_data:
```

### Docker 빌드 및 실행

```bash
# 이미지 빌드
docker build -t kuni-analysis:latest .

# 단독 실행
docker run -d -p 5000:5000 \
    -e DB_HOST=host.docker.internal \
    -e DB_USER=user \
    -e DB_PASSWORD=password \
    -e DB_NAME=database \
    -v $(pwd)/cache:/app/cache \
    kuni-analysis:latest

# Docker Compose 실행
docker-compose up -d
```

---

## 배포 자동화

### deploy.sh 스크립트

**파일**: [deploy.sh](../deploy.sh)

```bash
#!/bin/bash
set -e

echo "🚀 KUNI Analysis 배포 시작..."

PROJECT_DIR="/home/ubuntu/KUNI_2thecore_data_analysis"
SERVICE_NAME="kuni-analysis"

# 코드 업데이트
echo "📥 코드 업데이트..."
cd $PROJECT_DIR
git pull origin main

# 의존성 설치
echo "📦 의존성 설치..."
source venv/bin/activate
pip install -r requirements.txt

# 환경 검증
echo "🔍 환경 검증..."
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
```

### GitHub Actions CI/CD

`.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Server
        uses: appleboy/ssh-action@v0.1.10
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /home/ubuntu/KUNI_2thecore_data_analysis
            ./deploy.sh
```

---

## 모니터링

### 로그 확인

```bash
# 애플리케이션 로그
tail -f /var/log/kuni-analysis/error.log
tail -f /var/log/kuni-analysis/access.log

# Systemd 로그
journalctl -u kuni-analysis -f

# Nginx 로그
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### 헬스 체크

```bash
# API 헬스 체크
curl http://localhost:5000/api/health

# 응답 시간 측정
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/api/health
```

### 프로세스 모니터링

```bash
# Gunicorn 워커 상태
ps aux | grep gunicorn

# 메모리 사용량
free -h

# 디스크 사용량
df -h
```

---

## 보안 설정

### HTTPS 설정 (Let's Encrypt)

```bash
# Certbot 설치
sudo apt install certbot python3-certbot-nginx

# 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 설정
sudo certbot renew --dry-run
```

### 방화벽 설정

```bash
# UFW 활성화
sudo ufw enable

# 필요한 포트만 허용
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 상태 확인
sudo ufw status
```

### 환경 변수 보안

```bash
# .env 파일 권한 제한
chmod 600 .env

# root만 읽기 가능
chown root:root .env
```

---

## 문제 해결

### 한글 폰트 문제

차트에 한글이 표시되지 않는 경우:

```bash
# 폰트 설치 확인
fc-list :lang=ko

# 폰트 캐시 재생성
sudo fc-cache -fv

# matplotlib 캐시 삭제
rm -rf ~/.cache/matplotlib

# 서비스 재시작
sudo systemctl restart kuni-analysis
```

### 메모리 부족

```bash
# 스왑 파일 생성
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### DB 연결 실패

```bash
# MySQL 서비스 확인
sudo systemctl status mysql

# 연결 테스트
mysql -h $DB_HOST -u $DB_USER -p$DB_PASSWORD $DB_NAME -e "SELECT 1"

# Python에서 테스트
python -c "from src.data_loader import get_db_connection; print(get_db_connection())"
```

---

## 성능 튜닝

### Gunicorn 워커 수

```bash
# CPU 코어 수 * 2 + 1 권장
WORKERS=$(($(nproc) * 2 + 1))
gunicorn -w $WORKERS -b 0.0.0.0:5000 app:app
```

### 캐시 최적화

```python
# cache.py의 캐시 기간 조정
@cache_result(duration=3600)  # 1시간으로 증가
```

### 데이터베이스 최적화

```sql
-- 인덱스 추가
CREATE INDEX idx_drivelog_start_time ON drive_log(start_time);
CREATE INDEX idx_drivelog_car_id ON drive_log(car_id);
CREATE INDEX idx_car_brand ON car(brand);
```

---

**관련 문서**: [[Getting-Started]] | [[Architecture]] | [[API-Reference]]
