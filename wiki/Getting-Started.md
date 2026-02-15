# 시작하기

> KUNI 2thecore 데이터 분석 시스템 설치 및 설정 가이드

## 사전 요구사항

### 필수 소프트웨어
| 소프트웨어 | 버전 | 용도 |
|------------|------|------|
| Python | 3.9+ | 런타임 환경 |
| pip | 21.0+ | 패키지 관리 |
| MySQL | 5.7+ / 8.0+ | 데이터베이스 |
| Git | 2.0+ | 버전 관리 |

### 운영 체제별 준비

#### Windows
```powershell
# Python 설치 확인
python --version
pip --version

# Git 설치 확인
git --version
```

#### macOS
```bash
# Homebrew로 설치
brew install python@3.10 mysql git
```

#### Ubuntu/Linux
```bash
# 패키지 설치
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git mysql-client
```

---

## 1단계: 프로젝트 클론

```bash
# 저장소 클론
git clone https://github.com/your-org/KUNI_2thecore_data_analysis.git
cd KUNI_2thecore_data_analysis
```

---

## 2단계: 가상환경 설정

### Windows

```powershell
# 가상환경 생성
python -m venv .venv

# 활성화
.venv\Scripts\activate

# 비활성화 (나중에)
deactivate
```

### macOS / Linux

```bash
# 가상환경 생성
python3 -m venv .venv

# 활성화
source .venv/bin/activate

# 비활성화 (나중에)
deactivate
```

---

## 3단계: 의존성 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt
```

### 주요 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| Flask | 3.1.0 | 웹 프레임워크 |
| Flask-RESTful | 0.3.10 | REST API |
| flasgger | - | Swagger 문서화 |
| pandas | 2.3.2 | 데이터 처리 |
| numpy | 2.3.2 | 수치 연산 |
| scikit-learn | 1.7.1 | 머신러닝 |
| statsmodels | 0.14.5 | 시계열 분석 |
| matplotlib | 3.10.5 | 시각화 |
| seaborn | 0.13.2 | 시각화 |
| SQLAlchemy | 2.0.43 | ORM |
| mysql-connector-python | 9.4.0 | MySQL 드라이버 |

---

## 4단계: 환경 검증

```bash
# 라이브러리 임포트 테스트
python verify_setup.py
```

**예상 출력**:
```
--- 라이브러리 임포트 성공 ---
Pandas version: 2.3.2
NumPy version: 2.3.2
Scikit-learn version: 1.7.1
Matplotlib version: 3.10.5
Seaborn version: 0.13.2

--- Pandas 데이터프레임 생성 성공 ---
   A  B
0  1  a
1  2  b
2  3  c

환경 검증 완료. 주요 라이브러리가 모두 정상적으로 동작합니다.
```

---

## 5단계: 데이터베이스 설정

### .env 파일 생성

프로젝트 루트에 `.env` 파일을 생성합니다:

```env
DB_HOST=your_mysql_host
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=your_database_name
DB_PORT=3306
```

### 데이터베이스 스키마

#### car 테이블
```sql
CREATE TABLE car (
    car_id INT PRIMARY KEY AUTO_INCREMENT,
    model VARCHAR(100),
    brand VARCHAR(50),
    status VARCHAR(20) DEFAULT 'IDLE',
    car_year INT,
    car_type VARCHAR(20),
    car_number VARCHAR(20),
    sum_dist FLOAT DEFAULT 0,
    login_id VARCHAR(50),
    last_latitude FLOAT,
    last_longitude FLOAT
);
```

#### drive_log 테이블
```sql
CREATE TABLE drive_log (
    drive_log_id INT PRIMARY KEY AUTO_INCREMENT,
    car_id INT,
    drive_dist FLOAT,
    start_point VARCHAR(200),
    end_point VARCHAR(200),
    start_latitude FLOAT,
    start_longitude FLOAT,
    end_latitude FLOAT,
    end_longitude FLOAT,
    start_time DATETIME,
    end_time DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model VARCHAR(100),
    brand VARCHAR(50),
    memo TEXT,
    status VARCHAR(20),
    FOREIGN KEY (car_id) REFERENCES car(car_id)
);
```

### 연결 테스트

```bash
# 데이터베이스 연결 테스트
python src/data_loader.py
```

**예상 출력**:
```
============================== 데이터베이스 연결 테스트 ==============================
Available tables:
  Tables_in_your_database
0                      car
1                drive_log

데이터 조회 성공!
   car_id   model brand  ...
```

---

## 6단계: 서버 실행

### 개발 서버

```bash
# 방법 1: run_server.py 사용 (권장)
python run_server.py

# 방법 2: app.py 직접 실행
python app.py
```

**예상 출력**:
```
=== KUNI 2thecore Data Analysis Server ===
Starting Flask development server...
Server will be available at: http://localhost:5000
API endpoints:
  GET  /                          - API 정보
  POST /api/data                  - 데이터 쿼리 실행
  GET  /api/health                - 헬스 체크
  GET  /api/analysis/period       - 계절별/월별 선호도 분석
  GET  /api/analysis/trend        - 연도별 트렌드 분석
  GET  /api/forecast/daily        - 일별 운행량 예측
  GET  /api/clustering/regions    - 지역 클러스터링 분석
  GET  /apidocs/                  - Swagger API 문서
=====================================
 * Running on http://0.0.0.0:5000
```

### 접속 확인

- **API 기본 정보**: http://localhost:5000/
- **Swagger UI**: http://localhost:5000/apidocs/
- **헬스 체크**: http://localhost:5000/api/health

---

## 7단계: API 테스트

### cURL 사용

```bash
# 헬스 체크
curl http://localhost:5000/api/health

# 선호도 분석
curl "http://localhost:5000/api/analysis/period?year=2023&period_type=month"

# 트렌드 분석
curl "http://localhost:5000/api/analysis/trend?start_year=2020&end_year=2024"

# 일별 예측
curl "http://localhost:5000/api/forecast/daily?forecast_days=7"

# 클러스터링
curl "http://localhost:5000/api/clustering/regions?k=5"
```

### Python 사용

```python
import requests

BASE_URL = "http://localhost:5000"

# 헬스 체크
response = requests.get(f"{BASE_URL}/api/health")
print(response.json())

# 선호도 분석
response = requests.get(
    f"{BASE_URL}/api/analysis/period",
    params={"year": "2023", "period_type": "season"}
)
result = response.json()
print(f"분석 성공: {result['success']}")
```

### Swagger UI 사용

1. 브라우저에서 http://localhost:5000/apidocs/ 접속
2. 원하는 엔드포인트 선택
3. "Try it out" 클릭
4. 파라미터 입력 후 "Execute" 클릭
5. 응답 확인

---

## 한글 폰트 설정 (중요)

차트에 한글이 정상적으로 표시되려면 한글 폰트가 필요합니다.

### Windows
- 기본 제공 (Malgun Gothic)

### macOS
- 기본 제공 (Apple SD Gothic Neo)

### Ubuntu/Linux
```bash
# 한글 폰트 설치
sudo apt install -y fonts-noto-cjk fonts-nanum fonts-liberation

# 폰트 캐시 업데이트
sudo fc-cache -fv

# 설치 확인
fc-list :lang=ko
```

---

## 문제 해결

### 1. 의존성 설치 실패

```bash
# 캐시 삭제 후 재설치
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

### 2. MySQL 연결 실패

```bash
# 연결 정보 확인
cat .env

# MySQL 서버 상태 확인
mysql -h $DB_HOST -u $DB_USER -p -e "SELECT 1"
```

### 3. 포트 충돌

```bash
# 사용 중인 포트 확인
netstat -tulpn | grep 5000

# 다른 포트로 실행
python -c "from app import app; app.run(port=8080)"
```

### 4. 한글 깨짐

```bash
# matplotlib 캐시 삭제
rm -rf ~/.cache/matplotlib

# 폰트 재설정
python -c "from src.utils.font_config import setup_korean_font; setup_korean_font()"
```

---

## 다음 단계

- [[API-Reference]] - API 엔드포인트 상세 문서
- [[Architecture]] - 시스템 아키텍처 이해
- [[Deployment]] - 프로덕션 배포 가이드
- [[Module-Preference-Analysis]] - 선호도 분석 모듈 상세

---

## 지원

문제가 발생하면 다음을 확인하세요:

1. Python 버전: `python --version` (3.9 이상)
2. 가상환경 활성화 여부
3. `.env` 파일 존재 및 권한
4. MySQL 서버 접근 가능 여부
5. 한글 폰트 설치 여부

---

**관련 문서**: [[Deployment]] | [[API-Reference]] | [[Home]]
