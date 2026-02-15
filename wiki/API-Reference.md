# API 레퍼런스

> KUNI 2thecore 데이터 분석 REST API 완전 가이드

## 기본 정보

- **Base URL**: `http://localhost:5000`
- **Swagger UI**: `http://localhost:5000/apidocs/`
- **Content-Type**: `application/json`
- **응답 형식**: JSON

## 공통 응답 구조

### 성공 응답
```json
{
    "success": true,
    "message": "작업이 완료되었습니다.",
    "visualizations": {
        "chart_name": "data:image/jpeg;base64,..."
    },
    "data": { ... }
}
```

### 에러 응답
```json
{
    "success": false,
    "message": "오류 설명",
    "visualizations": {}
}
```

### HTTP 상태 코드
| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 (파라미터 오류) |
| 404 | 엔드포인트 없음 |
| 500 | 서버 내부 오류 |
| 503 | 서비스 불가 (DB 연결 실패) |

---

## 시스템 엔드포인트

### GET /
**설명**: API 기본 정보 및 사용 가능한 엔드포인트 목록을 반환합니다.

**소스 코드**: [app.py:37-62](../app.py#L37-L62)

**요청**:
```bash
curl http://localhost:5000/
```

**응답**:
```json
{
    "message": "KUNI 2thecore Data Analysis API",
    "status": "running",
    "endpoints": {
        "/": "API 정보",
        "/api/data": "데이터 조회 (POST)",
        "/api/health": "헬스 체크",
        "/api/analysis/period": "선호도 분석 (GET)",
        "/api/analysis/trend": "연도별 트렌드 분석 (GET)"
    }
}
```

---

### GET /api/health
**설명**: 서버 및 데이터베이스 연결 상태를 확인합니다.

**소스 코드**: [app.py:110-132](../app.py#L110-L132)

**요청**:
```bash
curl http://localhost:5000/api/health
```

**성공 응답** (200):
```json
{
    "status": "healthy",
    "database": "connected",
    "message": "시스템이 정상적으로 작동 중입니다."
}
```

**실패 응답** (503):
```json
{
    "status": "unhealthy",
    "database": "disconnected",
    "error": "Connection refused"
}
```

---

### POST /api/data
**설명**: 사용자 정의 SQL 쿼리를 실행합니다.

**소스 코드**: [app.py:65-107](../app.py#L65-L107)

**요청**:
```bash
curl -X POST http://localhost:5000/api/data \
     -H "Content-Type: application/json" \
     -d '{"query": "SELECT * FROM car LIMIT 5"}'
```

**요청 본문**:
```json
{
    "query": "SELECT * FROM car LIMIT 5"
}
```

**응답**:
```json
{
    "success": true,
    "data": [
        {
            "car_id": 1,
            "model": "K3",
            "brand": "기아",
            "status": "IDLE",
            "car_type": "소형"
        }
    ],
    "row_count": 5
}
```

**에러 응답** (400):
```json
{
    "error": "쿼리가 필요합니다. 'query' 필드를 포함해주세요."
}
```

---

## 분석 엔드포인트

### GET /api/analysis/period
**설명**: 월별 또는 계절별 브랜드 선호도를 분석합니다.

**소스 코드**: [app.py:170-182](../app.py#L170-L182)

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `year` | string | N | 전체 | 분석 대상 연도 |
| `period_type` | string | N | `month` | 기간 유형: `month` 또는 `season` |

**요청**:
```bash
# 2023년 월별 분석
curl "http://localhost:5000/api/analysis/period?year=2023&period_type=month"

# 전체 기간 계절별 분석
curl "http://localhost:5000/api/analysis/period?period_type=season"
```

**응답**:
```json
{
    "success": true,
    "message": "month 선호도 분석이 완료되었습니다.",
    "visualizations": {
        "brand_period_heatmap": "data:image/jpeg;base64,...",
        "market_share_pie": "data:image/jpeg;base64,...",
        "brand_preference_line": "data:image/jpeg;base64,...",
        "seasonality_strength_bar": "data:image/jpeg;base64,...",
        "statistical_comparison": "data:image/jpeg;base64,..."
    }
}
```

**시각화 항목**:
| 키 | 설명 |
|-----|------|
| `brand_period_heatmap` | 브랜드별 기간별 선호도 히트맵 |
| `market_share_pie` | 브랜드별 시장 점유율 파이차트 |
| `brand_preference_line` | 브랜드별 기간별 트렌드 라인차트 |
| `seasonality_strength_bar` | 브랜드별 계절성 강도 바차트 |
| `statistical_comparison` | 카이제곱 검정 결과 히트맵 |

---

### GET /api/analysis/trend
**설명**: 연도별 브랜드 및 차량 선호도 트렌드를 분석합니다.

**소스 코드**: [app.py:186-200](../app.py#L186-L200)

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `start_year` | int | N | 2020 | 분석 시작 연도 |
| `end_year` | int | N | 2025 | 분석 종료 연도 |
| `top_n` | int | N | 5 | 상위 모델 수 (1-20) |

**요청**:
```bash
curl "http://localhost:5000/api/analysis/trend?start_year=2020&end_year=2024&top_n=10"
```

**응답**:
```json
{
    "success": true,
    "message": "2020-2024 연도별 트렌드 분석이 완료되었습니다.",
    "visualizations": {
        "brand_trend_lines": "data:image/jpeg;base64,...",
        "model_ranking_change": "data:image/jpeg;base64,...",
        "car_age_preference": "data:image/jpeg;base64,...",
        "market_share_evolution": "data:image/jpeg;base64,...",
        "trend_summary": "data:image/jpeg;base64,..."
    }
}
```

**시각화 항목**:
| 키 | 설명 |
|-----|------|
| `brand_trend_lines` | 브랜드별 시장 점유율 트렌드 (회귀선 포함) |
| `model_ranking_change` | 상위 모델별 트렌드 변화 수평 바차트 |
| `car_age_preference` | 차량 연식별 선호도 트렌드 |
| `market_share_evolution` | 변동성 및 성장률 2-패널 차트 |
| `trend_summary` | 브랜드별 트렌드 요약 (방향, R²) |

---

### GET /api/forecast/daily
**설명**: SARIMA 모델을 사용하여 일별 운행 차량 수를 예측합니다.

**소스 코드**: [app.py:203-217](../app.py#L203-L217)

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `start_date` | string | N | 전체 | 분석 시작일 (YYYY-MM-DD) |
| `end_date` | string | N | 전체 | 분석 종료일 (YYYY-MM-DD) |
| `forecast_days` | int | N | 7 | 예측 기간 (1-30일) |

**요청**:
```bash
curl "http://localhost:5000/api/forecast/daily?forecast_days=14"

curl "http://localhost:5000/api/forecast/daily?start_date=2023-01-01&end_date=2023-12-31&forecast_days=7"
```

**응답**:
```json
{
    "success": true,
    "message": "일별 운행량 예측이 완료되었습니다.",
    "visualizations": {
        "usage_trend_with_prediction": "data:image/jpeg;base64,...",
        "weekday_pattern": "data:image/jpeg;base64,..."
    },
    "historical_data": [
        {
            "date": "2023-12-01",
            "unique_cars": 45,
            "total_trips": 120,
            "total_distance": 3500.5
        }
    ],
    "predictions": [
        {
            "date": "2024-01-01",
            "predicted_unique_cars": 42.3,
            "lower_ci": 35.1,
            "upper_ci": 49.5
        }
    ],
    "weekday_patterns": {
        "0": 38.5,
        "1": 42.3,
        "2": 40.1,
        "3": 41.8,
        "4": 43.2,
        "5": 35.6,
        "6": 28.9
    },
    "model_accuracy": {
        "method": "SARIMA",
        "sarima_order": [1, 1, 1],
        "seasonal_order": [1, 0, 1, 7],
        "aic": 1234.56,
        "bic": 1245.67,
        "mae": 3.45,
        "ljung_box_pvalue": 0.85
    }
}
```

---

### GET /api/clustering/regions
**설명**: K-means 또는 DBSCAN을 사용하여 지역별 수요를 클러스터링합니다.

**소스 코드**: [app.py:220-236](../app.py#L220-L236)

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `start_date` | string | N | 전체 | 분석 시작일 (YYYY-MM-DD) |
| `end_date` | string | N | 전체 | 분석 종료일 (YYYY-MM-DD) |
| `k` | int | N | 5 | K-means 클러스터 수 (1-50) |
| `use_end_points` | bool | N | true | 도착점 기준 분석 |
| `clustering_method` | string | N | kmeans | 알고리즘: `kmeans` 또는 `dbscan` |
| `min_trips` | int | N | 5 | 최소 운행 수 필터 |
| `eps_km` | float | N | 1.0 | DBSCAN eps (km) |
| `threshold_km` | float | N | 5.0 | 서비스 부족 판단 거리 (km) |

**요청**:
```bash
# K-means 클러스터링
curl "http://localhost:5000/api/clustering/regions?k=7"

# DBSCAN 클러스터링
curl "http://localhost:5000/api/clustering/regions?clustering_method=dbscan&eps_km=2.0"

# 출발점 기준 분석
curl "http://localhost:5000/api/clustering/regions?use_end_points=false"
```

**응답**:
```json
{
    "success": true,
    "message": "지역 클러스터링이 완료되었습니다.",
    "visualizations": {
        "cluster_map": "data:image/jpeg;base64,...",
        "recommendation_map": "data:image/jpeg;base64,..."
    },
    "cluster_summary": [
        {
            "cluster_id": 0,
            "trip_count": 523,
            "unique_cars": 45,
            "avg_distance": 12.5,
            "total_distance": 6537.5,
            "center_lat": 37.5665,
            "center_lng": 126.9780,
            "importance_score": 324.8
        }
    ],
    "importance_ranking": [
        { ... }
    ],
    "recommended_locations": [
        {
            "lat": 37.4820,
            "lon": 126.8970,
            "min_distance_km": 8.5
        }
    ],
    "underserved_areas": [ ... ],
    "current_coverage": {
        "covered_ratio": 0.72,
        "centers": [
            {
                "lat": 37.5665,
                "lon": 126.9780,
                "nearest_km": 2.3,
                "covered": true
            }
        ]
    }
}
```

---

## 에러 핸들링

### 500 Internal Server Error
**소스 코드**: [app.py:250-253](../app.py#L250-L253)

```json
{
    "error": "Internal server error"
}
```

### 404 Not Found
**소스 코드**: [app.py:255-257](../app.py#L255-L257)

```json
{
    "error": "Endpoint not found"
}
```

---

## 코드 예시

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:5000"

# 헬스 체크
response = requests.get(f"{BASE_URL}/api/health")
print(response.json())

# 선호도 분석
response = requests.get(
    f"{BASE_URL}/api/analysis/period",
    params={"year": "2023", "period_type": "month"}
)
result = response.json()

if result["success"]:
    # 이미지 저장
    import base64
    heatmap = result["visualizations"]["brand_period_heatmap"]
    image_data = heatmap.split(",")[1]
    with open("heatmap.jpg", "wb") as f:
        f.write(base64.b64decode(image_data))
```

### JavaScript (fetch)

```javascript
// 트렌드 분석
const response = await fetch(
    'http://localhost:5000/api/analysis/trend?start_year=2020&end_year=2024'
);
const data = await response.json();

if (data.success) {
    // 이미지를 img 태그에 표시
    document.getElementById('chart').src = data.visualizations.brand_trend_lines;
}
```

### cURL 스크립트

```bash
#!/bin/bash
BASE_URL="http://localhost:5000"

# 모든 분석 실행
echo "=== 헬스 체크 ==="
curl -s "$BASE_URL/api/health" | jq

echo "=== 선호도 분석 ==="
curl -s "$BASE_URL/api/analysis/period?year=2023" | jq '.success, .message'

echo "=== 트렌드 분석 ==="
curl -s "$BASE_URL/api/analysis/trend" | jq '.success, .message'

echo "=== 일별 예측 ==="
curl -s "$BASE_URL/api/forecast/daily?forecast_days=7" | jq '.success, .model_accuracy'

echo "=== 클러스터링 ==="
curl -s "$BASE_URL/api/clustering/regions?k=5" | jq '.success, .current_coverage.covered_ratio'
```

---

## 성능 고려사항

### 캐싱
- 분석 결과는 **30분(1800초)** 동안 캐싱됩니다.
- 동일 파라미터 요청 시 캐시된 결과가 반환됩니다.
- 캐시 파일: `/cache/*.pkl`

### 응답 시간
| 엔드포인트 | 평균 응답 시간 (캐시 미스) |
|------------|---------------------------|
| `/api/health` | < 100ms |
| `/api/analysis/period` | 2-5초 |
| `/api/analysis/trend` | 3-8초 |
| `/api/forecast/daily` | 5-15초 |
| `/api/clustering/regions` | 3-10초 |

### 이미지 최적화
- 포맷: JPEG
- DPI: 75
- 인코딩: base64

---

**관련 문서**: [[Architecture]] | [[Data-Flow]] | [[Deployment]]
