# 데이터 흐름

> KUNI 2thecore 데이터 분석 시스템의 요청 처리 흐름 및 데이터 파이프라인

## 전체 데이터 흐름 개요

```mermaid
flowchart TB
    subgraph "입력"
        A[HTTP Request]
        B[Query Parameters]
    end

    subgraph "API 레이어"
        C[Flask Router]
        D[BaseAnalysisAPI.get_param]
        E[BaseAnalysisAPI.execute_analysis]
    end

    subgraph "서비스 레이어"
        F[Analyzer.__init__]
        G[Analyzer.analyze]
        H[Cache Check]
    end

    subgraph "데이터 처리"
        I[_load_data]
        J[SQL Query]
        K[DataFrame 처리]
    end

    subgraph "분석 & 시각화"
        L[통계 분석]
        M[ML 알고리즘]
        N[차트 생성]
        O[base64 인코딩]
    end

    subgraph "출력"
        P[JSON Response]
    end

    A --> C
    B --> D
    C --> E
    D --> E
    E --> F --> G
    G --> H
    H -->|Cache Hit| P
    H -->|Cache Miss| I
    I --> J --> K
    K --> L & M
    L & M --> N --> O --> P
```

## 요청 처리 시퀀스

### 1. 선호도 분석 요청 흐름

```mermaid
sequenceDiagram
    participant Client
    participant Flask as Flask Router
    participant API as PreferenceAnalysisAPI
    participant Base as BaseAnalysisAPI
    participant Analyzer as SimplePreferenceAnalyzer
    participant Cache as CacheManager
    participant DB as MySQL Database
    participant Viz as Matplotlib

    Client->>Flask: GET /api/analysis/period?year=2023&period_type=month
    Flask->>API: get()
    API->>Base: get_param('year')
    API->>Base: get_param('period_type', 'month')
    Base-->>API: year='2023', period_type='month'

    API->>Base: execute_analysis('SimplePreferenceAnalyzer', ...)
    Base->>Analyzer: __import__() + getattr()
    Base->>Analyzer: analyze_preferences(year='2023', period_type='month')

    Analyzer->>Cache: @cache_result 체크
    alt Cache Hit
        Cache-->>Analyzer: 캐시된 결과 반환
    else Cache Miss
        Analyzer->>DB: SELECT dl.start_time, dl.brand, dl.model, c.car_type...
        DB-->>Analyzer: DataFrame
        Analyzer->>Analyzer: 데이터 전처리 (year, month, season 추출)
        Analyzer->>Viz: _create_heatmap()
        Analyzer->>Viz: _create_pie_chart()
        Analyzer->>Viz: _create_line_chart()
        Viz-->>Analyzer: base64 이미지들
        Analyzer->>Cache: 결과 저장
    end

    Analyzer-->>Base: {"success": true, "visualizations": {...}}
    Base-->>API: 결과
    API-->>Flask: JSON Response
    Flask-->>Client: HTTP 200 + JSON
```

### 2. 일별 예측 요청 흐름

```mermaid
sequenceDiagram
    participant Client
    participant API as DailyForecastAPI
    participant Analyzer as DailyForecastAnalyzer
    participant DB as MySQL
    participant SARIMA as statsmodels.SARIMAX

    Client->>API: GET /api/forecast/daily?forecast_days=7
    API->>Analyzer: analyze(forecast_days=7)

    Analyzer->>DB: SELECT dl.start_time, dl.car_id, dl.drive_log_id, dl.drive_dist...
    DB-->>Analyzer: DataFrame

    Analyzer->>Analyzer: 일별 통계 집계 (groupby 'date')
    Note over Analyzer: unique_cars, total_trips, total_distance

    Analyzer->>Analyzer: _preprocess_timeseries()
    Note over Analyzer: 정상성 검정 (ADF Test)

    Analyzer->>Analyzer: _find_best_sarima_params()
    Note over Analyzer: AIC 기준 최적 파라미터 탐색

    Analyzer->>SARIMA: SARIMAX(series, order, seasonal_order)
    SARIMA->>SARIMA: fit(disp=False)
    SARIMA->>SARIMA: forecast(steps=7)
    SARIMA-->>Analyzer: 예측값 + 신뢰구간

    Analyzer->>Analyzer: _plot_usage_with_prediction()
    Analyzer->>Analyzer: _plot_weekday_pattern()

    Analyzer-->>API: {"success": true, "predictions": [...], "model_accuracy": {...}}
    API-->>Client: JSON Response
```

### 3. 지역 클러스터링 요청 흐름

```mermaid
sequenceDiagram
    participant Client
    participant API as RegionClusteringAPI
    participant Analyzer as RegionClusteringAnalyzer
    participant DB as MySQL
    participant KMeans as sklearn.KMeans

    Client->>API: GET /api/clustering/regions?k=5
    API->>Analyzer: analyze(k=5, use_end_points=true)

    Analyzer->>DB: SELECT end_latitude AS lat, end_longitude AS lon...
    DB-->>Analyzer: 좌표 DataFrame

    Analyzer->>Analyzer: 좌표 유효성 검증
    Note over Analyzer: NaN, 0 좌표 제거

    Analyzer->>KMeans: KMeans(n_clusters=5)
    KMeans->>KMeans: fit_predict(coords)
    KMeans-->>Analyzer: cluster labels + centers

    Analyzer->>Analyzer: _summarize_clusters()
    Note over Analyzer: trip_count, unique_cars, importance_score 계산

    Analyzer->>DB: SELECT last_latitude, last_longitude FROM car
    DB-->>Analyzer: 현재 차량 위치

    Analyzer->>Analyzer: _find_underserved_areas()
    Note over Analyzer: Haversine 거리 계산

    Analyzer->>Analyzer: _compute_current_coverage()

    Analyzer-->>API: {"success": true, "cluster_summary": [...], "underserved_areas": [...]}
    API-->>Client: JSON Response
```

## 데이터 처리 파이프라인

### 선호도 분석 파이프라인

```mermaid
flowchart LR
    subgraph "데이터 로드"
        A[SQL Query] --> B[drive_log + car JOIN]
        B --> C[DataFrame]
    end

    subgraph "전처리"
        C --> D[날짜 파싱]
        D --> E[year, month 추출]
        E --> F[season 매핑]
        F --> G[결측값 제거]
    end

    subgraph "분석"
        G --> H[crosstab 생성]
        H --> I[Chi-square 검정]
        G --> J[그룹별 집계]
        J --> K[계절성 강도 계산]
    end

    subgraph "시각화"
        H --> L[히트맵]
        J --> M[라인차트]
        G --> N[파이차트]
        K --> O[바차트]
    end
```

### SARIMA 예측 파이프라인

```mermaid
flowchart LR
    subgraph "데이터 준비"
        A[drive_log 조회] --> B[일별 집계]
        B --> C[시계열 생성]
        C --> D[결측 날짜 채우기]
    end

    subgraph "전처리"
        D --> E[영점 처리]
        E --> F[ADF 정상성 검정]
        F -->|비정상| G[차분]
        F -->|정상| H[원본 유지]
    end

    subgraph "모델링"
        G --> I[파라미터 탐색]
        H --> I
        I --> J[SARIMA 학습]
        J --> K[예측 생성]
        K --> L[신뢰구간 계산]
    end

    subgraph "평가"
        J --> M[AIC/BIC]
        J --> N[MAE 계산]
        J --> O[Ljung-Box 검정]
    end
```

### K-means 클러스터링 파이프라인

```mermaid
flowchart LR
    subgraph "데이터 수집"
        A[end_latitude, end_longitude] --> B[좌표 추출]
        B --> C[유효성 검증]
    end

    subgraph "클러스터링"
        C --> D[KMeans 초기화]
        D --> E[fit_predict]
        E --> F[중심점 계산]
        E --> G[라벨 할당]
    end

    subgraph "분석"
        G --> H[클러스터별 집계]
        H --> I[중요도 점수]
        F --> J[차량 위치 비교]
        J --> K[Haversine 거리]
        K --> L[부족 지역 식별]
    end
```

## 데이터베이스 쿼리 패턴

### 선호도 분석 쿼리
**위치**: [simple_preference_analysis.py:95-103](../src/simple_preference_analysis.py#L95-L103)

```sql
SELECT dl.start_time, dl.brand, dl.model, c.car_type
FROM drive_log dl
JOIN car c ON dl.car_id = c.car_id
WHERE dl.start_time IS NOT NULL
  AND YEAR(dl.start_time) = {year}  -- 선택적
```

### 트렌드 분석 쿼리
**위치**: [simple_trend_analysis.py:72-84](../src/simple_trend_analysis.py#L72-L84)

```sql
SELECT
    YEAR(dl.start_time) as drive_year,
    dl.brand,
    dl.model,
    c.car_year,
    COUNT(*) as drive_count
FROM drive_log dl
JOIN car c ON dl.car_id = c.car_id
WHERE YEAR(dl.start_time) BETWEEN {start_year} AND {end_year}
GROUP BY YEAR(dl.start_time), dl.brand, dl.model, c.car_year
ORDER BY drive_year, dl.brand
```

### 일별 예측 쿼리
**위치**: [daily_forecast.py:106-114](../src/services/daily_forecast.py#L106-L114)

```sql
SELECT
    dl.start_time,
    dl.car_id,
    dl.drive_log_id,
    dl.drive_dist
FROM drive_log dl
WHERE dl.start_time IS NOT NULL
  AND DATE(dl.start_time) >= '{start_date}'
  AND DATE(dl.start_time) <= '{end_date}'
```

### 클러스터링 쿼리 (도착점 기준)
**위치**: [region_clustering.py:118-126](../src/services/region_clustering.py#L118-L126)

```sql
SELECT
    dl.end_latitude AS lat,
    dl.end_longitude AS lon,
    dl.start_latitude,
    dl.start_longitude,
    dl.start_point,
    dl.end_point,
    dl.drive_dist,
    dl.car_id
FROM drive_log dl
WHERE dl.start_time IS NOT NULL
  AND dl.end_latitude IS NOT NULL
  AND dl.end_longitude IS NOT NULL
```

## 응답 데이터 형식

### 성공 응답 구조
```json
{
    "success": true,
    "message": "분석이 완료되었습니다.",
    "visualizations": {
        "chart_name": "data:image/jpeg;base64,/9j/4AAQSkZ..."
    },
    "data": { ... }  // 분석별 추가 데이터
}
```

### 시각화 데이터 흐름
```mermaid
flowchart LR
    A[matplotlib Figure] --> B[savefig to BytesIO]
    B --> C[buffer.seek(0)]
    C --> D[base64.b64encode]
    D --> E["data:image/jpeg;base64,{encoded}"]
    E --> F[JSON Response]
```

**코드 참조**: [simple_preference_analysis.py:290-298](../src/simple_preference_analysis.py#L290-L298)

```python
def _fig_to_base64(self, fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='jpeg', dpi=75, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/jpeg;base64,{image_base64}"
```

## 캐싱 메커니즘

### 캐시 키 생성
**위치**: [cache.py:12-14](../src/utils/cache.py#L12-L14)

```python
cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
# 예: "analyze_5842674930928"
```

### 캐시 흐름
```mermaid
flowchart TD
    A[분석 요청] --> B{캐시 파일 존재?}
    B -->|No| C[분석 실행]
    B -->|Yes| D{캐시 만료?}
    D -->|Yes| C
    D -->|No| E[캐시 로드]
    C --> F[결과 저장]
    F --> G[응답 반환]
    E --> G
```

### 캐시 파일 구조
```
cache/
├── analyze_5842674930928.pkl      # 선호도 분석 결과
├── analyze_yearly_trend_...pkl    # 트렌드 분석 결과
├── analyze_1234567890.pkl         # 일별 예측 결과
└── analyze_9876543210.pkl         # 클러스터링 결과
```

## 에러 처리 흐름

```mermaid
flowchart TD
    A[요청 수신] --> B{파라미터 유효성}
    B -->|Invalid| C[400 Bad Request]
    B -->|Valid| D{DB 연결}
    D -->|실패| E[503 Service Unavailable]
    D -->|성공| F{분석 실행}
    F -->|예외 발생| G[500 Internal Error]
    F -->|성공| H[200 OK + JSON]
    G --> I[로그 기록]
```

### 에러 응답 형식
```json
{
    "success": false,
    "message": "분석 중 오류가 발생했습니다: {error_detail}",
    "visualizations": {}
}
```

---

**관련 문서**: [[Architecture]] | [[API-Reference]] | [[Diagrams]]
