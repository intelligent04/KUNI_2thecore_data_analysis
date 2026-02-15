# 다이어그램 모음

> KUNI 2thecore 데이터 분석 시스템의 Mermaid 다이어그램 컬렉션

## 시스템 아키텍처

### 전체 시스템 구조

```mermaid
graph TB
    subgraph "클라이언트 레이어"
        WEB[웹 브라우저]
        API_CLIENT[API 클라이언트<br/>curl, Postman]
        SWAGGER[Swagger UI<br/>/apidocs/]
    end

    subgraph "애플리케이션 레이어"
        subgraph "Flask 서버"
            FLASK[Flask App<br/>app.py:16]
            CORS[CORS 미들웨어]
            RESTFUL[Flask-RESTful<br/>app.py:34]
        end

        subgraph "API 리소스"
            DATA_API[DataAnalysisAPI<br/>GET /]
            QUERY_API[DataQueryAPI<br/>POST /api/data]
            HEALTH_API[HealthCheckAPI<br/>GET /api/health]
            PREF_API[PreferenceAnalysisAPI<br/>GET /api/analysis/period]
            TREND_API[TrendAnalysisAPI<br/>GET /api/analysis/trend]
            FORECAST_API[DailyForecastAPI<br/>GET /api/forecast/daily]
            CLUSTER_API[RegionClusteringAPI<br/>GET /api/clustering/regions]
        end
    end

    subgraph "서비스 레이어"
        PREF_SVC[SimplePreferenceAnalyzer]
        TREND_SVC[SimpleTrendAnalyzer]
        FORECAST_SVC[DailyForecastAnalyzer]
        CLUSTER_SVC[RegionClusteringAnalyzer]
    end

    subgraph "데이터 레이어"
        LOADER[DataLoader<br/>data_loader.py]
        CACHE[CacheManager<br/>cache.py]
    end

    subgraph "외부 시스템"
        MYSQL[(MySQL Database)]
        FS[File System<br/>/cache]
    end

    WEB & API_CLIENT & SWAGGER --> FLASK
    FLASK --> CORS --> RESTFUL
    RESTFUL --> DATA_API & QUERY_API & HEALTH_API
    RESTFUL --> PREF_API & TREND_API & FORECAST_API & CLUSTER_API

    PREF_API --> PREF_SVC
    TREND_API --> TREND_SVC
    FORECAST_API --> FORECAST_SVC
    CLUSTER_API --> CLUSTER_SVC

    PREF_SVC & TREND_SVC & FORECAST_SVC & CLUSTER_SVC --> LOADER
    PREF_SVC & TREND_SVC & FORECAST_SVC & CLUSTER_SVC --> CACHE

    LOADER --> MYSQL
    CACHE --> FS
```

### 컴포넌트 관계도

```mermaid
graph LR
    subgraph "진입점"
        A[app.py]
        B[run_server.py]
    end

    subgraph "분석 모듈"
        C[simple_preference_analysis.py]
        D[simple_trend_analysis.py]
        E[daily_forecast.py]
        F[region_clustering.py]
    end

    subgraph "유틸리티"
        G[data_loader.py]
        H[cache.py]
        I[font_config.py]
    end

    subgraph "외부 라이브러리"
        J[Flask/Flask-RESTful]
        K[pandas/numpy]
        L[sklearn]
        M[statsmodels]
        N[matplotlib/seaborn]
        O[SQLAlchemy]
    end

    B --> A
    A --> J
    A --> C & D & E & F

    C --> G & H & I & K & L & N
    D --> G & H & I & K & L & N
    E --> G & H & I & K & M & N
    F --> G & H & I & K & L & N

    G --> O & K
```

---

## 클래스 다이어그램

### API 클래스 계층 구조

```mermaid
classDiagram
    class Resource {
        <<Flask-RESTful>>
    }

    class BaseAnalysisAPI {
        +execute_analysis(analyzer_class, module, method, kwargs) Dict
        +get_param(name, default, type) Any
    }

    class DataAnalysisAPI {
        +get() Dict
    }

    class DataQueryAPI {
        +post() Dict
    }

    class HealthCheckAPI {
        +get() Dict
    }

    class PreferenceAnalysisAPI {
        +get() Dict
    }

    class TrendAnalysisAPI {
        +get() Dict
    }

    class DailyForecastAPI {
        +get() Dict
    }

    class RegionClusteringAPI {
        +get() Dict
    }

    Resource <|-- BaseAnalysisAPI
    Resource <|-- DataAnalysisAPI
    Resource <|-- DataQueryAPI
    Resource <|-- HealthCheckAPI
    BaseAnalysisAPI <|-- PreferenceAnalysisAPI
    BaseAnalysisAPI <|-- TrendAnalysisAPI
    BaseAnalysisAPI <|-- DailyForecastAPI
    BaseAnalysisAPI <|-- RegionClusteringAPI
```

### Analyzer 클래스 구조

```mermaid
classDiagram
    class SimplePreferenceAnalyzer {
        -brand_colors: Dict~str, str~
        -dynamic_colors: Dict~str, str~
        -color_palette: List~str~
        -season_names: Dict~int, str~
        +analyze_preferences(year, period_type) Dict
        -_load_data(year) DataFrame
        -_create_all_charts(df, period_type) Dict
        -_create_heatmap(df, period_type) str
        -_create_pie_chart(df) str
        -_create_line_chart(df, period_type) str
        -_create_seasonality_chart(df) str
        -_create_statistical_chart(df, period_type) str
        -_color_for_brand(brand) str
        -_fig_to_base64(fig) str
    }

    class SimpleTrendAnalyzer {
        -brand_colors: Dict~str, str~
        -current_year: int
        +analyze_yearly_trend(start_year, end_year, top_n) Dict
        -_load_trend_data(start, end) DataFrame
        -_analyze_trends(df, top_n) Dict
        -_analyze_brand_trends(df) Dict
        -_analyze_model_trends(df, n) Dict
        -_analyze_car_age_trends(df) Dict
        -_analyze_market_share_evolution(df) Dict
        -_create_trend_charts(df, results, n) Dict
    }

    class DailyForecastAnalyzer {
        -max_forecast_days: int
        +analyze(start_date, end_date, forecast_days) Dict
        -_load_data(start, end) DataFrame
        -_forecast_sarima(stats, days) Tuple
        -_preprocess_timeseries(series) Tuple
        -_find_best_sarima_params(series) Tuple
        -_fallback_arima_forecast(series, days) Tuple
        -_plot_usage_with_prediction(stats, forecast) str
        -_plot_weekday_pattern(pattern) str
    }

    class RegionClusteringAnalyzer {
        -default_k: int
        +analyze(start, end, k, use_end, method, min_trips, eps, threshold) Dict
        -_load_data(start, end, use_end) DataFrame
        -_compute_dbscan_centers(df) DataFrame
        -_summarize_clusters(df, centers, min) Tuple
        -_load_car_locations() List
        -_haversine_km(lat1, lon1, lat2, lon2) float
        -_find_underserved_areas(centers, cars, threshold) List
        -_compute_current_coverage(centers, cars, threshold) Dict
        -_plot_clusters(df, centers) str
        -_plot_recommendations(centers, underserved) str
    }
```

---

## 시퀀스 다이어그램

### 선호도 분석 요청 흐름

```mermaid
sequenceDiagram
    participant Client
    participant Flask
    participant PreferenceAPI
    participant BaseAPI
    participant Analyzer
    participant Cache
    participant DB
    participant Matplotlib

    Client->>Flask: GET /api/analysis/period?year=2023
    Flask->>PreferenceAPI: get()
    PreferenceAPI->>BaseAPI: get_param('year')
    BaseAPI-->>PreferenceAPI: '2023'
    PreferenceAPI->>BaseAPI: execute_analysis(...)

    BaseAPI->>Analyzer: analyze_preferences(year='2023')
    Analyzer->>Cache: @cache_result 체크

    alt 캐시 히트
        Cache-->>Analyzer: 캐시된 결과
    else 캐시 미스
        Analyzer->>DB: SQL 쿼리
        DB-->>Analyzer: DataFrame

        Analyzer->>Analyzer: 데이터 전처리
        Analyzer->>Matplotlib: 차트 생성
        Matplotlib-->>Analyzer: Figure

        Analyzer->>Analyzer: _fig_to_base64()
        Analyzer->>Cache: 결과 저장
    end

    Analyzer-->>BaseAPI: 결과 Dict
    BaseAPI-->>PreferenceAPI: 결과
    PreferenceAPI-->>Flask: JSON
    Flask-->>Client: HTTP 200
```

### SARIMA 예측 흐름

```mermaid
sequenceDiagram
    participant API as DailyForecastAPI
    participant Analyzer as DailyForecastAnalyzer
    participant Preprocessor as 전처리
    participant ParamSearch as 파라미터 탐색
    participant SARIMA as SARIMAX

    API->>Analyzer: analyze(forecast_days=7)

    Analyzer->>Analyzer: _load_data()
    Note over Analyzer: SQL 쿼리 실행

    Analyzer->>Analyzer: 일별 집계
    Note over Analyzer: groupby('date')

    Analyzer->>Preprocessor: _preprocess_timeseries()
    Preprocessor->>Preprocessor: ADF 정상성 검정
    alt 비정상
        Preprocessor->>Preprocessor: 차분 수행
    end
    Preprocessor-->>Analyzer: 전처리된 시계열

    Analyzer->>ParamSearch: _find_best_sarima_params()
    loop p, d, q 조합
        ParamSearch->>SARIMA: SARIMAX.fit()
        SARIMA-->>ParamSearch: AIC
    end
    ParamSearch-->>Analyzer: 최적 파라미터

    Analyzer->>SARIMA: SARIMAX(최적 파라미터)
    SARIMA->>SARIMA: fit(disp=False)
    SARIMA->>SARIMA: forecast(steps=7)
    SARIMA-->>Analyzer: 예측값 + 신뢰구간

    Analyzer-->>API: 결과 Dict
```

### K-means 클러스터링 흐름

```mermaid
sequenceDiagram
    participant API as RegionClusteringAPI
    participant Analyzer as RegionClusteringAnalyzer
    participant KMeans as sklearn.KMeans
    participant Haversine as 거리 계산

    API->>Analyzer: analyze(k=5)

    Analyzer->>Analyzer: _load_data()
    Note over Analyzer: 좌표 유효성 검증

    Analyzer->>KMeans: KMeans(n_clusters=5)
    KMeans->>KMeans: fit_predict(coords)
    KMeans-->>Analyzer: labels + centers

    Analyzer->>Analyzer: _summarize_clusters()
    Note over Analyzer: 중요도 점수 계산

    Analyzer->>Analyzer: _load_car_locations()
    Note over Analyzer: 현재 차량 위치

    loop 각 수요 중심점
        Analyzer->>Haversine: _haversine_km()
        Haversine-->>Analyzer: 거리 (km)
    end

    Analyzer->>Analyzer: _find_underserved_areas()
    Analyzer->>Analyzer: _compute_current_coverage()

    Analyzer-->>API: 결과 Dict
```

---

## 데이터 흐름 다이어그램

### 전체 데이터 파이프라인

```mermaid
flowchart LR
    subgraph "데이터 소스"
        A[(MySQL)]
    end

    subgraph "데이터 로딩"
        B[get_db_connection]
        C[get_data_from_db]
        D[pandas.read_sql]
    end

    subgraph "전처리"
        E[날짜 파싱]
        F[파생 변수 생성]
        G[결측값 처리]
    end

    subgraph "분석"
        H[통계 분석]
        I[ML 모델링]
        J[시계열 예측]
    end

    subgraph "시각화"
        K[matplotlib Figure]
        L[base64 인코딩]
    end

    subgraph "출력"
        M[JSON Response]
    end

    A --> B --> C --> D
    D --> E --> F --> G
    G --> H & I & J
    H & I & J --> K --> L --> M
```

### 캐시 메커니즘

```mermaid
flowchart TD
    A[분석 요청] --> B{캐시 키 생성}
    B --> C{파일 존재?}

    C -->|Yes| D{만료 체크}
    C -->|No| E[분석 실행]

    D -->|만료됨| E
    D -->|유효| F[캐시 로드]

    E --> G[결과 생성]
    G --> H[캐시 저장]
    H --> I[응답 반환]
    F --> I

    style F fill:#90EE90
    style H fill:#FFB6C1
```

---

## 데이터베이스 ER 다이어그램

```mermaid
erDiagram
    CAR {
        int car_id PK
        varchar model
        varchar brand
        varchar status
        int car_year
        varchar car_type
        varchar car_number
        float sum_dist
        varchar login_id
        float last_latitude
        float last_longitude
    }

    DRIVE_LOG {
        int drive_log_id PK
        int car_id FK
        float drive_dist
        varchar start_point
        varchar end_point
        float start_latitude
        float start_longitude
        float end_latitude
        float end_longitude
        datetime start_time
        datetime end_time
        datetime created_at
        varchar model
        varchar brand
        varchar memo
        varchar status
    }

    CAR ||--o{ DRIVE_LOG : "has many"
```

---

## 배포 아키텍처

### 개발 환경

```mermaid
graph TB
    subgraph "개발 머신"
        DEV[개발자]
        IDE[VS Code / PyCharm]
        VENV[Python venv]
    end

    subgraph "로컬 서버"
        FLASK[Flask 개발 서버<br/>:5000]
        SWAGGER[Swagger UI<br/>/apidocs/]
    end

    subgraph "로컬 DB"
        MYSQL[(MySQL<br/>:3306)]
    end

    DEV --> IDE --> VENV
    VENV --> FLASK
    FLASK --> SWAGGER
    FLASK --> MYSQL
```

### 프로덕션 환경

```mermaid
graph TB
    subgraph "인터넷"
        USERS[사용자들]
    end

    subgraph "웹 서버"
        NGINX[Nginx<br/>:80/:443]
    end

    subgraph "애플리케이션 서버"
        GUNICORN[Gunicorn<br/>4 workers]
        FLASK1[Flask Worker 1]
        FLASK2[Flask Worker 2]
        FLASK3[Flask Worker 3]
        FLASK4[Flask Worker 4]
    end

    subgraph "데이터 레이어"
        MYSQL[(MySQL<br/>:3306)]
        CACHE[(Redis 캐시<br/>권장)]
    end

    USERS --> NGINX
    NGINX --> GUNICORN
    GUNICORN --> FLASK1 & FLASK2 & FLASK3 & FLASK4
    FLASK1 & FLASK2 & FLASK3 & FLASK4 --> MYSQL
    FLASK1 & FLASK2 & FLASK3 & FLASK4 -.-> CACHE
```

---

## 상태 다이어그램

### 분석 요청 상태

```mermaid
stateDiagram-v2
    [*] --> Received: HTTP 요청 수신
    Received --> Validating: 파라미터 검증

    Validating --> Invalid: 검증 실패
    Invalid --> [*]: 400 응답

    Validating --> CacheCheck: 검증 성공
    CacheCheck --> CacheHit: 캐시 존재

    CacheHit --> [*]: 캐시 응답

    CacheCheck --> Processing: 캐시 미스
    Processing --> LoadingData: 데이터 로드
    LoadingData --> Analyzing: 분석 수행
    Analyzing --> Visualizing: 시각화 생성
    Visualizing --> Caching: 결과 캐싱
    Caching --> [*]: 200 응답

    Processing --> Error: 예외 발생
    Error --> [*]: 500 응답
```

---

## 사용법

### Mermaid 렌더링

GitHub Wiki는 Mermaid 다이어그램을 자동으로 렌더링합니다. 아래와 같이 코드 블록을 사용하세요:

````markdown
```mermaid
graph LR
    A --> B
```
````

### 로컬 렌더링

Mermaid CLI를 사용하여 이미지로 변환할 수 있습니다:

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i diagram.mmd -o diagram.png
```

---

**관련 문서**: [[Architecture]] | [[Data-Flow]] | [[API-Reference]]
