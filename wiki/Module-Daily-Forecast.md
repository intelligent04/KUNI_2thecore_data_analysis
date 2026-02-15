# 일별 운행량 예측 모듈

> SARIMA 시계열 모델 기반 일별 운행 차량 수 예측

## 개요

일별 운행량 예측 모듈은 과거 운행 데이터를 기반으로 SARIMA(Seasonal ARIMA) 모델을 사용하여 향후 7~30일간의 일별 운행 차량 수를 예측합니다.

**소스 파일**: [src/services/daily_forecast.py](../src/services/daily_forecast.py)

## 클래스 구조

### DailyForecastAnalyzer

```python
class DailyForecastAnalyzer:
    def __init__(self):
        self.max_forecast_days = 30  # 최대 예측 기간
```

## API 엔드포인트

### GET /api/forecast/daily

일별 운행량 분석 및 예측을 수행합니다.

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `start_date` | string | 아니오 | 전체 | 분석 시작일 (YYYY-MM-DD) |
| `end_date` | string | 아니오 | 전체 | 분석 종료일 (YYYY-MM-DD) |
| `forecast_days` | int | 아니오 | 7 | 예측 기간 (1-30일) |

**요청 예시**:
```bash
# 최근 데이터 기반 7일 예측
curl "http://localhost:5000/api/forecast/daily?forecast_days=7"

# 특정 기간 데이터로 14일 예측
curl "http://localhost:5000/api/forecast/daily?start_date=2023-01-01&end_date=2023-12-31&forecast_days=14"
```

**응답 예시**:
```json
{
    "success": true,
    "message": "일별 운행량 예측이 완료되었습니다.",
    "visualizations": {
        "usage_trend_with_prediction": "data:image/jpeg;base64,...",
        "weekday_pattern": "data:image/jpeg;base64,..."
    },
    "historical_data": [
        {"date": "2023-12-01", "unique_cars": 45, "total_trips": 120, "total_distance": 3500.5},
        ...
    ],
    "predictions": [
        {"date": "2024-01-01", "predicted_unique_cars": 42.3, "lower_ci": 35.1, "upper_ci": 49.5},
        ...
    ],
    "weekday_patterns": {
        "0": 38.5,  // 월요일
        "1": 42.3,  // 화요일
        ...
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

## 메서드 상세

### analyze(start_date, end_date, forecast_days)
**위치**: [daily_forecast.py:37-93](../src/services/daily_forecast.py#L37-L93)

메인 분석 및 예측 함수입니다.

```python
@cache_result(duration=1800)
def analyze(self, start_date: str = None, end_date: str = None,
            forecast_days: int = 7) -> Dict[str, Any]:
    """
    Args:
        start_date: 분석 시작일 (YYYY-MM-DD)
        end_date: 분석 종료일 (YYYY-MM-DD)
        forecast_days: 예측 기간 (1-30일)

    Returns:
        {
            "success": bool,
            "visualizations": Dict,
            "historical_data": List,
            "predictions": List,
            "weekday_patterns": Dict,
            "model_accuracy": Dict
        }
    """
```

### _load_data(start_date, end_date)
**위치**: [daily_forecast.py:95-122](../src/services/daily_forecast.py#L95-L122)

운행 로그 데이터를 로드합니다.

```python
def _load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
    query = f"""
    SELECT
        dl.start_time,
        dl.car_id,
        dl.drive_log_id,
        dl.drive_dist
    FROM drive_log dl
    WHERE dl.start_time IS NOT NULL
      AND DATE(dl.start_time) >= '{start_date}'
      AND DATE(dl.start_time) <= '{end_date}'
    """

    df = get_data_from_db(query)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['date'] = df['start_time'].dt.date
    return df[['date', 'car_id', 'drive_log_id', 'drive_dist']]
```

## SARIMA 예측 파이프라인

### 1. 시계열 전처리
**위치**: [daily_forecast.py:196-223](../src/services/daily_forecast.py#L196-L223)

```python
def _preprocess_timeseries(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """시계열 전처리: 정상성 확인 및 변환"""

    # 영점 처리 (log 변환을 위해)
    if (series <= 0).any():
        series = series + 1

    # 정상성 검정 (ADF test)
    adf_result = adfuller(series.dropna())
    is_stationary = adf_result[1] < 0.05

    # 비정상 시계열이면 차분 수행
    if not is_stationary:
        series_diff = series.diff().dropna()
        # 차분 후 재검정
        ...
```

### 2. 최적 파라미터 탐색
**위치**: [daily_forecast.py:225-269](../src/services/daily_forecast.py#L225-L269)

AIC(Akaike Information Criterion) 기준으로 최적 SARIMA 파라미터를 탐색합니다.

```python
def _find_best_sarima_params(self, series: pd.Series) -> Tuple[Tuple, Tuple]:
    """SARIMA 최적 파라미터 탐색 (AIC 기준)"""

    # 데이터 길이에 따른 파라미터 범위 조정
    max_p = min(3, len(series) // 10)
    max_d = min(2, len(series) // 20)
    max_q = min(3, len(series) // 10)

    # 계절성 주기 (주간 패턴)
    seasonal_period = 7 if len(series) >= 21 else 0

    best_aic = np.inf
    best_params = (1, 1, 1)
    best_seasonal_params = (0, 0, 0, 0)

    # 그리드 서치
    for p, d, q in product(range(max_p+1), range(max_d+1), range(max_q+1)):
        if p + d + q == 0:
            continue

        try:
            model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_params)
            fitted = model.fit(disp=False, maxiter=100)

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_params = (p, d, q)
                best_seasonal_params = seasonal_params
        except:
            continue

    return best_params, best_seasonal_params
```

### 3. SARIMA 예측 수행
**위치**: [daily_forecast.py:124-194](../src/services/daily_forecast.py#L124-L194)

```python
def _forecast_sarima(self, daily_stats: pd.DataFrame, forecast_days: int):
    # 시계열 준비
    full_dates = pd.date_range(daily_stats['date'].min(), daily_stats['date'].max())
    series = daily_stats.set_index(pd.to_datetime(daily_stats['date']))['unique_cars']
    series = series.reindex(full_dates, fill_value=0)

    # 전처리
    series_processed, transformation_info = self._preprocess_timeseries(series)

    # 최소 데이터 체크 (2주 이상 필요)
    if len(series_processed) < 14:
        return self._mean_fallback_forecast(series, forecast_days)

    # 최적 파라미터 탐색
    best_params, best_seasonal_params = self._find_best_sarima_params(series_processed)

    # SARIMA 모델 학습
    model = SARIMAX(
        series_processed,
        order=best_params,
        seasonal_order=best_seasonal_params,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)

    # 예측 수행
    forecast = fitted_model.forecast(steps=forecast_days)
    forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()

    # 음수 값 보정
    forecast = np.maximum(forecast, 0)
```

### 4. 폴백 메커니즘
**위치**: [daily_forecast.py:271-308](../src/services/daily_forecast.py#L271-L308)

SARIMA 모델 학습 실패 시 단순 ARIMA 또는 평균값으로 대체합니다.

```python
def _fallback_arima_forecast(self, series: pd.Series, forecast_days: int):
    """SARIMA 실패시 단순 ARIMA 대체 모델"""
    try:
        model = ARIMA(series, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_days)
        return forecast_df, {'method': 'ARIMA_fallback', ...}
    except:
        # 최종 대체: 단순 평균
        mean_value = series.mean()
        return mean_forecast_df, {'method': 'mean_fallback', ...}
```

## 모델 평가 지표

### SARIMA 모델 지표
| 지표 | 설명 | 좋은 값 |
|------|------|---------|
| `aic` | Akaike Information Criterion | 낮을수록 좋음 |
| `bic` | Bayesian Information Criterion | 낮을수록 좋음 |
| `mae` | Mean Absolute Error | 낮을수록 좋음 |
| `ljung_box_pvalue` | 잔차 자기상관 검정 | > 0.05 (자기상관 없음) |

### SARIMA 파라미터
- **order (p, d, q)**: 비계절 ARIMA 파라미터
  - p: 자기회귀 차수
  - d: 차분 차수
  - q: 이동평균 차수
- **seasonal_order (P, D, Q, s)**: 계절 파라미터
  - s: 계절 주기 (7 = 주간)

## 시각화 상세

### 1. 운행량 트렌드 및 예측 차트
**위치**: [daily_forecast.py:310-348](../src/services/daily_forecast.py#L310-L348)

```python
def _plot_usage_with_prediction(self, daily_stats, forecast_df) -> str:
    fig, ax = plt.subplots(figsize=(14, 8))

    # 실제 데이터
    ax.plot(actual_dates, daily_stats['unique_cars'],
            label='실제 운행 차량 수', color='#1f77b4', marker='o')

    # 예측 데이터
    ax.plot(forecast_dates, forecast_df['predicted_unique_cars'],
            label='SARIMA 예측', color='#ff7f0e', linestyle='--', marker='s')

    # 95% 신뢰구간
    ax.fill_between(forecast_dates,
                    forecast_df['lower_ci'],
                    forecast_df['upper_ci'],
                    color='#ff7f0e', alpha=0.2, label='95% 신뢰구간')
```

### 2. 요일별 패턴 차트
**위치**: [daily_forecast.py:350-360](../src/services/daily_forecast.py#L350-L360)

```python
def _plot_weekday_pattern(self, weekday_pattern: Dict[int, float]) -> str:
    labels = ['월', '화', '수', '목', '금', '토', '일']
    ax.bar([labels[w] for w in weekdays], values, color='#2ca02c')
    ax.set_title('요일별 평균 운행 차량 수')
```

## 데이터 집계

### 일별 통계
```python
daily_stats = df.groupby('date').agg(
    unique_cars=('car_id', 'nunique'),     # 고유 차량 수
    total_trips=('drive_log_id', 'count'),  # 총 운행 건수
    total_distance=('drive_dist', 'sum')    # 총 운행 거리
).reset_index()
```

### 요일별 패턴
```python
weekday_pattern = tmp.groupby('weekday')['unique_cars'].mean().to_dict()
# {0: 월요일 평균, 1: 화요일 평균, ..., 6: 일요일 평균}
```

## 사용 예시

### Python 코드에서 직접 사용

```python
from src.services.daily_forecast import DailyForecastAnalyzer

analyzer = DailyForecastAnalyzer()

# 14일 예측
result = analyzer.analyze(
    start_date='2023-01-01',
    end_date='2023-12-31',
    forecast_days=14
)

if result['success']:
    # 예측 결과 확인
    for pred in result['predictions']:
        print(f"{pred['date']}: {pred['predicted_unique_cars']:.1f} "
              f"({pred['lower_ci']:.1f} - {pred['upper_ci']:.1f})")

    # 모델 정확도 확인
    accuracy = result['model_accuracy']
    print(f"Method: {accuracy['method']}")
    print(f"MAE: {accuracy['mae']:.2f}")
    print(f"AIC: {accuracy['aic']:.2f}")
```

### 예측 결과 해석

```json
{
    "predictions": [
        {
            "date": "2024-01-01",
            "predicted_unique_cars": 42.3,
            "lower_ci": 35.1,
            "upper_ci": 49.5
        }
    ]
}
```
- `predicted_unique_cars`: 예측 운행 차량 수
- `lower_ci`: 95% 신뢰구간 하한
- `upper_ci`: 95% 신뢰구간 상한

## 의존성

| 패키지 | 용도 |
|--------|------|
| pandas | 데이터 처리 |
| numpy | 수치 연산 |
| statsmodels.tsa.statespace.sarimax | SARIMAX 모델 |
| statsmodels.tsa.arima.model | ARIMA 폴백 |
| statsmodels.tsa.stattools | ADF 검정 |
| statsmodels.stats.diagnostic | Ljung-Box 검정 |
| sklearn.metrics | MAE 계산 |
| matplotlib | 시각화 |

## 제한사항

1. **최소 데이터 요구**: 2주(14일) 이상의 데이터 필요
2. **최대 예측 기간**: 30일
3. **계절성**: 7일(주간) 주기만 고려
4. **캐싱**: 30분간 동일 파라미터 결과 캐싱

---

**관련 문서**: [[Module-Region-Clustering]] | [[Module-Trend-Analysis]] | [[API-Reference]]
