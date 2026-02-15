# 트렌드 분석 모듈

> 연도별 브랜드 및 차량 선호도 변화 추세 분석

## 개요

트렌드 분석 모듈은 다년간의 렌터카 운행 데이터를 분석하여 브랜드별 시장 점유율 변화, 모델별 선호도 추이, 차량 연식별 선호도 패턴을 파악합니다.

**소스 파일**: [src/simple_trend_analysis.py](../src/simple_trend_analysis.py)

## 클래스 구조

### SimpleTrendAnalyzer

```python
class SimpleTrendAnalyzer:
    """간소화된 트렌드 분석 클래스"""

    def __init__(self):
        self.brand_colors = {'현대': '#1f77b4', '기아': '#ff7f0e', '제네시스': '#2ca02c'}
        self.current_year = datetime.now().year
```

## API 엔드포인트

### GET /api/analysis/trend

연도별 브랜드 및 차량 선호도 트렌드를 분석합니다.

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `start_year` | int | 아니오 | 2020 | 분석 시작 연도 |
| `end_year` | int | 아니오 | 2025 | 분석 종료 연도 |
| `top_n` | int | 아니오 | 5 | 상위 모델 수 (1-20) |

**요청 예시**:
```bash
# 2020-2024년, 상위 10개 모델 분석
curl "http://localhost:5000/api/analysis/trend?start_year=2020&end_year=2024&top_n=10"
```

**응답 예시**:
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

## 메서드 상세

### analyze_yearly_trend(start_year, end_year, top_n)
**위치**: [simple_trend_analysis.py:39-66](../src/simple_trend_analysis.py#L39-L66)

메인 트렌드 분석 함수입니다.

```python
def analyze_yearly_trend(self, start_year: int = 2020, end_year: int = 2025,
                         top_n: int = 5) -> Dict[str, Any]:
    """
    Args:
        start_year: 분석 시작 연도
        end_year: 분석 종료 연도
        top_n: 상위 모델 개수

    Returns:
        분석 결과 및 시각화 딕셔너리
    """
    df = self._load_trend_data(start_year, end_year)
    trend_results = self._analyze_trends(df, top_n)
    visualizations = self._create_trend_charts(df, trend_results, top_n)
```

### _load_trend_data(start_year, end_year)
**위치**: [simple_trend_analysis.py:68-98](../src/simple_trend_analysis.py#L68-L98)

연도별 집계된 운행 데이터를 로드합니다.

```python
def _load_trend_data(self, start_year: int, end_year: int) -> pd.DataFrame:
    query = f"""
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
    """

    df = get_data_from_db(query)

    # 파생 변수 계산
    df['car_age'] = df['drive_year'] - df['car_year']  # 차량 연식

    # 연도별 시장 점유율 계산
    yearly_totals = df.groupby('drive_year')['drive_count'].sum()
    df['market_share'] = df['drive_count'] / df['drive_year'].map(yearly_totals)
```

## 분석 항목

### 1. 브랜드 트렌드 분석
**위치**: [simple_trend_analysis.py:118-151](../src/simple_trend_analysis.py#L118-L151)

선형회귀를 사용하여 브랜드별 시장 점유율 추세를 분석합니다.

```python
def _analyze_brand_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
    for brand in df['brand'].unique():
        brand_data = df[df['brand'] == brand].groupby('drive_year')['market_share'].sum()

        # sklearn 선형회귀
        X = brand_data.index.values.reshape(-1, 1)
        y = brand_data.values

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        r2 = r2_score(y, model.predict(X))

        # 트렌드 방향 분류
        if slope > 0.01:
            trend_direction = 'increasing'
        elif slope < -0.01:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
```

**출력**:
```json
{
    "현대": {
        "slope": 0.015,
        "r2_score": 0.85,
        "trend_direction": "increasing",
        "yearly_data": {2020: 0.42, 2021: 0.44, ...}
    }
}
```

### 2. 상위 모델 트렌드 분석
**위치**: [simple_trend_analysis.py:153-185](../src/simple_trend_analysis.py#L153-L185)

전체 기간 운행량 기준 상위 N개 모델의 트렌드를 분석합니다.

```python
def _analyze_model_trends(self, df: pd.DataFrame, top_n: int) -> Dict[str, Any]:
    # 전체 기간 상위 모델 선별
    model_totals = df.groupby(['brand', 'model'])['drive_count'].sum()
    top_models = model_totals.sort_values(ascending=False).head(top_n).index

    for brand, model in top_models:
        # 모델별 선형회귀 분석
        ...
```

### 3. 차량 연식별 선호도 분석
**위치**: [simple_trend_analysis.py:187-219](../src/simple_trend_analysis.py#L187-L219)

차량 연식을 카테고리화하여 연령대별 선호도 변화를 분석합니다.

```python
def _analyze_car_age_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
    # 연령대별 분류
    df_age = df.copy()
    df_age['age_category'] = pd.cut(
        df_age['car_age'],
        bins=[-float('inf'), 2, 5, 10, float('inf')],
        labels=['신차(0-2년)', '준신차(3-5년)', '중고차(6-10년)', '노후차(11년+)']
    )
```

**카테고리**:
| 카테고리 | 차량 연식 |
|----------|-----------|
| 신차 | 0-2년 |
| 준신차 | 3-5년 |
| 중고차 | 6-10년 |
| 노후차 | 11년 이상 |

### 4. 시장 점유율 진화 분석
**위치**: [simple_trend_analysis.py:221-244](../src/simple_trend_analysis.py#L221-L244)

브랜드별 시장 점유율 변동성과 성장률을 계산합니다.

```python
def _analyze_market_share_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
    yearly_brand_share = df.groupby(['drive_year', 'brand'])['market_share'].sum().unstack()

    for brand in yearly_brand_share.columns:
        shares = yearly_brand_share[brand].values

        # 변동성 (변동계수)
        volatility = np.std(shares) / np.mean(shares)

        # 성장률 (첫해 대비 마지막해)
        growth_rate = (shares[-1] - shares[0]) / shares[0]
```

## 시각화 상세

### 1. 브랜드별 트렌드 라인
**위치**: [simple_trend_analysis.py:283-311](../src/simple_trend_analysis.py#L283-L311)

```python
def _create_brand_trend_chart(self, brand_trends: Dict[str, Any]) -> str:
    for brand, data in brand_trends.items():
        years = list(data['yearly_data'].keys())
        shares = list(data['yearly_data'].values())

        # 실제 데이터 플롯
        ax.plot(years, shares, marker='o', linewidth=2.5,
                label=f"{brand} ({data['trend_direction']})")

        # 트렌드 라인 (점선)
        X = np.array(years).reshape(-1, 1)
        model = LinearRegression().fit(X, shares)
        trend_line = model.predict(X)
        ax.plot(years, trend_line, linestyle='--', alpha=0.7)
```

### 2. 모델 랭킹 변화 차트
**위치**: [simple_trend_analysis.py:313-350](../src/simple_trend_analysis.py#L313-L350)

수평 바 차트로 상위 모델의 트렌드 방향과 강도를 표시합니다.

```python
def _create_model_ranking_chart(self, model_trends: Dict[str, Any]) -> str:
    for i, (model_key, data) in enumerate(top_5_models):
        trend_direction = data['trend_direction']
        slope = data['slope']

        # 트렌드 방향별 색상
        color = 'green' if trend_direction == 'increasing' else \
                'red' if trend_direction == 'decreasing' else 'gray'
        symbol = '↗' if trend_direction == 'increasing' else \
                 '↘' if trend_direction == 'decreasing' else '→'

        ax.barh(i, abs(slope) * 100, color=color, alpha=0.7)
```

### 3. 차량 연식별 선호도 차트
**위치**: [simple_trend_analysis.py:352-376](../src/simple_trend_analysis.py#L352-L376)

연령대별 트렌드 기울기를 바 차트로 표시합니다.

### 4. 시장 진화 차트 (2-패널)
**위치**: [simple_trend_analysis.py:378-411](../src/simple_trend_analysis.py#L378-L411)

```python
def _create_market_evolution_chart(self, market_evolution: Dict[str, Any]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 왼쪽: 변동성 차트
    ax1.bar(brands, vol_values, color=colors)
    ax1.set_title('브랜드별 시장점유율 변동성')

    # 오른쪽: 성장률 차트
    ax2.bar(brands_growth, growth_values, color=colors_growth)
    ax2.set_title('브랜드별 점유율 성장률')
```

### 5. 트렌드 요약 차트
**위치**: [simple_trend_analysis.py:433-474](../src/simple_trend_analysis.py#L433-L474)

모든 브랜드의 트렌드 방향과 R² 점수를 요약합니다.

## 통계 지표

### 선형회귀 지표
| 지표 | 설명 | 해석 |
|------|------|------|
| `slope` | 기울기 | 연간 시장 점유율 변화량 |
| `r2_score` | 결정계수 | 모델 적합도 (0~1) |
| `trend_direction` | 트렌드 방향 | increasing/decreasing/stable |

### 트렌드 방향 기준
```python
if slope > 0.01:
    trend_direction = 'increasing'   # 연 1%p 이상 증가
elif slope < -0.01:
    trend_direction = 'decreasing'   # 연 1%p 이상 감소
else:
    trend_direction = 'stable'       # 안정적
```

### 변동성 지표
- **변동계수 (CV)**: 표준편차 / 평균
- 높은 CV = 시장 점유율 변동이 큼

### 성장률
- **(최근년도 점유율 - 첫해 점유율) / 첫해 점유율**
- 양수 = 성장, 음수 = 하락

## 사용 예시

### Python 코드에서 직접 사용

```python
from src.simple_trend_analysis import SimpleTrendAnalyzer

analyzer = SimpleTrendAnalyzer()

# 2020-2024년 트렌드 분석
result = analyzer.analyze_yearly_trend(
    start_year=2020,
    end_year=2024,
    top_n=10
)

if result['success']:
    # 브랜드 트렌드 차트
    trend_chart = result['visualizations']['brand_trend_lines']

    # 브랜드별 트렌드 방향 확인
    for brand, data in result.get('brand_trends', {}).items():
        print(f"{brand}: {data['trend_direction']} (R²={data['r2_score']:.2f})")
```

### Flask API 사용

```python
# app.py:186-200
class TrendAnalysisAPI(BaseAnalysisAPI):
    def get(self):
        start_year = self.get_param('start_year', 2020, int)
        end_year = self.get_param('end_year', 2025, int)
        top_n = self.get_param('top_n', 5, int)

        return self.execute_analysis(
            'SimpleTrendAnalyzer',
            'src.simple_trend_analysis',
            'analyze_yearly_trend',
            start_year=start_year,
            end_year=end_year,
            top_n=top_n
        )
```

## 의존성

| 패키지 | 용도 |
|--------|------|
| pandas | 데이터 처리 및 그룹화 |
| numpy | 수치 연산 |
| sklearn.linear_model | LinearRegression |
| sklearn.metrics | r2_score |
| matplotlib | 차트 생성 |

---

**관련 문서**: [[Module-Preference-Analysis]] | [[Module-Daily-Forecast]] | [[API-Reference]]
