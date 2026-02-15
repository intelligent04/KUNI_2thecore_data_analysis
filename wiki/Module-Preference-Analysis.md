# 선호도 분석 모듈

> 월별/계절별 브랜드 및 차량 선호도 패턴 분석

## 개요

선호도 분석 모듈은 렌터카 운행 데이터를 기반으로 브랜드, 모델, 차량 유형별 선호도를 월별 및 계절별로 분석합니다.

**소스 파일**: [src/simple_preference_analysis.py](../src/simple_preference_analysis.py)

## 클래스 구조

### SimplePreferenceAnalyzer

```python
class SimplePreferenceAnalyzer:
    """간소화된 선호도 분석 클래스"""

    def __init__(self):
        self.brand_colors = {'현대': '#1f77b4', '기아': '#ff7f0e', '제네시스': '#2ca02c'}
        self.season_names = {1: '봄', 2: '여름', 3: '가을', 4: '겨울'}
```

## API 엔드포인트

### GET /api/analysis/period

월별 또는 계절별 브랜드 선호도를 분석합니다.

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `year` | string | 아니오 | 전체 | 분석 대상 연도 (예: 2023) |
| `period_type` | string | 아니오 | `month` | 기간 유형: `month` 또는 `season` |

**요청 예시**:
```bash
# 2023년 월별 분석
curl "http://localhost:5000/api/analysis/period?year=2023&period_type=month"

# 전체 기간 계절별 분석
curl "http://localhost:5000/api/analysis/period?period_type=season"
```

**응답 예시**:
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

## 메서드 상세

### analyze_preferences(year, period_type)
**위치**: [simple_preference_analysis.py:66-89](../src/simple_preference_analysis.py#L66-L89)

메인 분석 함수로, 전체 분석 프로세스를 조율합니다.

```python
def analyze_preferences(self, year: Optional[str] = None,
                        period_type: str = 'month') -> Dict[str, Any]:
    """
    Args:
        year: 분석 대상 연도 (None이면 전체 기간)
        period_type: 'month' 또는 'season'

    Returns:
        {
            "success": bool,
            "message": str,
            "visualizations": Dict[str, str]  # base64 인코딩된 이미지들
        }
    """
```

### _load_data(year)
**위치**: [simple_preference_analysis.py:91-117](../src/simple_preference_analysis.py#L91-L117)

데이터베이스에서 운행 데이터를 로드하고 전처리합니다.

```python
def _load_data(self, year: Optional[str]) -> pd.DataFrame:
    query = """
    SELECT dl.start_time, dl.brand, dl.model, c.car_type
    FROM drive_log dl
    JOIN car c ON dl.car_id = c.car_id
    WHERE dl.start_time IS NOT NULL
    """
    # 연도 필터 추가
    if year:
        query += f" AND YEAR(dl.start_time) = {year}"

    df = get_data_from_db(query)

    # 전처리
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['year'] = df['start_time'].dt.year
    df['month'] = df['start_time'].dt.month
    df['season'] = df['month'].map({
        12: 4, 1: 4, 2: 4,   # 겨울
        3: 1, 4: 1, 5: 1,    # 봄
        6: 2, 7: 2, 8: 2,    # 여름
        9: 3, 10: 3, 11: 3   # 가을
    })
    return df.dropna()
```

### _create_all_charts(df, period_type)
**위치**: [simple_preference_analysis.py:119-153](../src/simple_preference_analysis.py#L119-L153)

모든 시각화 차트를 생성합니다.

| 차트 | 메서드 | 설명 |
|------|--------|------|
| `brand_period_heatmap` | `_create_heatmap()` | 브랜드별 기간별 히트맵 |
| `market_share_pie` | `_create_pie_chart()` | 시장 점유율 파이차트 |
| `brand_preference_line` | `_create_line_chart()` | 트렌드 라인차트 |
| `seasonality_strength_bar` | `_create_seasonality_chart()` | 계절성 강도 바차트 |
| `statistical_comparison` | `_create_statistical_chart()` | 카이제곱 검정 결과 |

## 시각화 상세

### 1. 브랜드별 기간별 히트맵
**위치**: [simple_preference_analysis.py:155-168](../src/simple_preference_analysis.py#L155-L168)

```python
def _create_heatmap(self, df: pd.DataFrame, period_type: str) -> str:
    period_col = 'month' if period_type == 'month' else 'season'
    crosstab = pd.crosstab(df['brand'], df[period_col], normalize='columns')

    plt, sns = _get_mpl()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt='.2f', cmap='YlGn', ax=ax)
```

**출력 예시**:
- X축: 월(1-12) 또는 계절(봄, 여름, 가을, 겨울)
- Y축: 브랜드 (현대, 기아, 제네시스 등)
- 값: 정규화된 선호도 비율 (0.00 ~ 1.00)

### 2. 시장 점유율 파이차트
**위치**: [simple_preference_analysis.py:170-183](../src/simple_preference_analysis.py#L170-L183)

```python
def _create_pie_chart(self, df: pd.DataFrame) -> str:
    brand_counts = df['brand'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [self._color_for_brand(b) for b in brand_counts.index]

    ax.pie(brand_counts.values, labels=brand_counts.index,
           autopct='%1.1f%%', colors=colors, startangle=90)
```

### 3. 계절성 강도 분석
**위치**: [simple_preference_analysis.py:213-249](../src/simple_preference_analysis.py#L213-L249)

계절성 강도는 변동계수(Coefficient of Variation)로 측정합니다:

```python
def _create_seasonality_chart(self, df: pd.DataFrame) -> str:
    seasonality_scores = {}

    for brand in df['brand'].unique():
        brand_data = df[df['brand'] == brand]
        seasonal_counts = brand_data.groupby('season').size()

        if len(seasonal_counts) > 1:
            cv = seasonal_counts.std() / seasonal_counts.mean()
            seasonality_scores[brand] = cv
```

**해석**:
- CV > 0.3: 높은 계절성 (특정 계절에 수요 집중)
- CV < 0.1: 낮은 계절성 (연중 고른 수요)

### 4. 통계적 검정 (카이제곱)
**위치**: [simple_preference_analysis.py:251-287](../src/simple_preference_analysis.py#L251-L287)

브랜드와 기간 간의 연관성을 카이제곱 검정으로 분석합니다.

```python
def _create_statistical_chart(self, df: pd.DataFrame, period_type: str) -> str:
    crosstab = pd.crosstab(df['brand'], df[period_col])

    # Chi-square test
    chi2_res = chi2_contingency(crosstab.to_numpy())
    chi2, p_value, dof, expected_arr = chi2_res

    significance = "통계적으로 유의함" if p_value < 0.05 else "통계적으로 유의하지 않음"
```

**출력**:
- 왼쪽: 관측값 히트맵 (χ² 값 표시)
- 오른쪽: 기댓값 히트맵 (p-value 표시)

## 색상 관리

### 브랜드별 고정 색상
**위치**: [simple_preference_analysis.py:35](../src/simple_preference_analysis.py#L35)

```python
brand_colors = {
    '현대': '#1f77b4',    # 파란색
    '기아': '#ff7f0e',    # 주황색
    '제네시스': '#2ca02c'  # 녹색
}
```

### 동적 색상 할당
**위치**: [simple_preference_analysis.py:43-64](../src/simple_preference_analysis.py#L43-L64)

새로운 브랜드가 발견되면 팔레트에서 순환 할당합니다.

```python
def _color_for_brand(self, brand: str) -> str:
    if brand in self.brand_colors:
        return self.brand_colors[brand]
    if brand in self.dynamic_colors:
        return self.dynamic_colors[brand]

    color = self.color_palette[self.palette_idx % len(self.color_palette)]
    self.palette_idx += 1
    self.dynamic_colors[brand] = color
    return color
```

## 테스트

### 단위 테스트
**위치**: [src/tests/test_simple_preference_analysis.py](../src/tests/test_simple_preference_analysis.py)

```python
class TestSimplePreferenceAnalyzer(unittest.TestCase):

    def test_load_data(self, mock_get_data_from_db):
        """데이터 로드 및 전처리 테스트"""
        mock_get_data_from_db.return_value = self.sample_df
        df = self.analyzer._load_data(year='2023')

        self.assertIn('year', df.columns)
        self.assertIn('month', df.columns)
        self.assertIn('season', df.columns)

    def test_create_all_charts(self):
        """차트 생성 테스트"""
        charts = self.analyzer._create_all_charts(self.sample_df, 'month')

        for chart_name, chart_data in charts.items():
            self.assertTrue(chart_data.startswith('data:image/jpeg;base64,'))

    def test_color_for_brand(self):
        """브랜드 색상 할당 테스트"""
        self.assertEqual(self.analyzer._color_for_brand('현대'), '#1f77b4')
        dynamic_color = self.analyzer._color_for_brand('르노')
        self.assertEqual(self.analyzer._color_for_brand('르노'), dynamic_color)
```

## 사용 예시

### Python 코드에서 직접 사용

```python
from src.simple_preference_analysis import SimplePreferenceAnalyzer

analyzer = SimplePreferenceAnalyzer()

# 2023년 월별 분석
result = analyzer.analyze_preferences(year='2023', period_type='month')

if result['success']:
    # 히트맵 이미지 추출
    heatmap_base64 = result['visualizations']['brand_period_heatmap']

    # base64 디코딩하여 파일로 저장
    import base64
    image_data = heatmap_base64.split(',')[1]
    with open('heatmap.jpg', 'wb') as f:
        f.write(base64.b64decode(image_data))
```

### API 호출

```bash
# cURL
curl -X GET "http://localhost:5000/api/analysis/period?year=2023&period_type=season" \
     -H "Accept: application/json"

# Python requests
import requests
response = requests.get(
    "http://localhost:5000/api/analysis/period",
    params={"year": "2023", "period_type": "month"}
)
data = response.json()
```

## 의존성

| 패키지 | 용도 |
|--------|------|
| pandas | 데이터 처리 및 crosstab |
| numpy | 수치 연산 |
| scipy.stats | chi2_contingency 검정 |
| sklearn.preprocessing | LabelEncoder |
| matplotlib | 차트 생성 |
| seaborn | 히트맵 스타일링 |

---

**관련 문서**: [[Module-Trend-Analysis]] | [[API-Reference]] | [[Architecture]]
