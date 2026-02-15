# 지역 클러스터링 모듈

> K-means/DBSCAN 기반 최적 렌터카 위치 분석

## 개요

지역 클러스터링 모듈은 렌터카 운행 위치 데이터를 분석하여 수요가 집중된 지역을 식별하고, 신규 렌터카 위치 설정을 위한 추천을 제공합니다.

**소스 파일**: [src/services/region_clustering.py](../src/services/region_clustering.py)

## 클래스 구조

### RegionClusteringAnalyzer

```python
class RegionClusteringAnalyzer:
    def __init__(self):
        self.default_k = 5  # 기본 클러스터 수
```

## API 엔드포인트

### GET /api/clustering/regions

지역별 수요 클러스터링 및 최적 위치 분석을 수행합니다.

**쿼리 파라미터**:
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `start_date` | string | 아니오 | 전체 | 분석 시작일 (YYYY-MM-DD) |
| `end_date` | string | 아니오 | 전체 | 분석 종료일 (YYYY-MM-DD) |
| `k` | int | 아니오 | 5 | K-means 클러스터 수 (1-50) |
| `use_end_points` | bool | 아니오 | true | 도착점 기준 분석 여부 |
| `clustering_method` | string | 아니오 | kmeans | 알고리즘: `kmeans` 또는 `dbscan` |
| `min_trips` | int | 아니오 | 5 | 최소 운행 수 필터 |
| `eps_km` | float | 아니오 | 1.0 | DBSCAN eps 파라미터 (km) |
| `threshold_km` | float | 아니오 | 5.0 | 서비스 부족 판단 거리 (km) |

**요청 예시**:
```bash
# K-means 5개 클러스터
curl "http://localhost:5000/api/clustering/regions?k=5"

# DBSCAN 클러스터링
curl "http://localhost:5000/api/clustering/regions?clustering_method=dbscan&eps_km=2.0"

# 출발점 기준 분석
curl "http://localhost:5000/api/clustering/regions?use_end_points=false"
```

**응답 예시**:
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
    "importance_ranking": [...],
    "recommended_locations": [
        {"lat": 37.4820, "lon": 126.8970, "min_distance_km": 8.5}
    ],
    "underserved_areas": [...],
    "current_coverage": {
        "covered_ratio": 0.72,
        "centers": [
            {"lat": 37.5665, "lon": 126.9780, "nearest_km": 2.3, "covered": true}
        ]
    }
}
```

## 메서드 상세

### analyze(start_date, end_date, k, use_end_points, method, min_trips, eps_km, threshold_km)
**위치**: [region_clustering.py:29-105](../src/services/region_clustering.py#L29-L105)

메인 클러스터링 분석 함수입니다.

```python
@cache_result(duration=1800)
def analyze(self, start_date: str = None, end_date: str = None,
            k: int = 5, use_end_points: bool = True,
            method: str = 'kmeans', min_trips: int = 5,
            eps_km: float = 1.0, threshold_km: float = 5.0) -> Dict[str, Any]:
```

### _load_data(start_date, end_date, use_end_points)
**위치**: [region_clustering.py:107-147](../src/services/region_clustering.py#L107-L147)

좌표 데이터를 로드하고 유효성을 검증합니다.

```python
def _load_data(self, start_date: str, end_date: str, use_end_points: bool) -> pd.DataFrame:
    if use_end_points:
        query = f"""
        SELECT
            dl.end_latitude AS lat, dl.end_longitude AS lon,
            dl.start_point, dl.end_point,
            dl.drive_dist, dl.car_id
        FROM drive_log dl
        WHERE dl.end_latitude IS NOT NULL AND dl.end_longitude IS NOT NULL
        """
    else:
        query = f"""
        SELECT
            dl.start_latitude AS lat, dl.start_longitude AS lon,
            ...
        """

    df = get_data_from_db(query)

    # 좌표 유효성 검증
    coords = coords[(coords['lat'] != 0) & (coords['lon'] != 0)]
    coords = coords[np.isfinite(coords['lat']) & np.isfinite(coords['lon'])]
```

## 클러스터링 알고리즘

### K-means 클러스터링
**위치**: [region_clustering.py:64-72](../src/services/region_clustering.py#L64-L72)

```python
if method.lower() == 'kmeans':
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(coords.values)
    centers = pd.DataFrame(model.cluster_centers_, columns=['lat', 'lon'])
```

**특징**:
- 지정된 k개의 클러스터 생성
- 클러스터 중심점 직접 계산
- 구형 클러스터에 적합

### DBSCAN 클러스터링
**위치**: [region_clustering.py:57-63](../src/services/region_clustering.py#L57-L63)

```python
if method.lower() == 'dbscan':
    eps_deg = eps_km / 111.0  # km를 위도/경도 단위로 변환
    model = DBSCAN(eps=eps_deg, min_samples=max(3, min_trips))
    labels = model.fit_predict(coords.values)
```

**특징**:
- 밀도 기반 클러스터링
- 클러스터 수 자동 결정
- 노이즈 포인트 식별 (label = -1)
- 불규칙한 형태의 클러스터 감지 가능

## 분석 결과

### 클러스터 요약
**위치**: [region_clustering.py:156-183](../src/services/region_clustering.py#L156-L183)

```python
def _summarize_clusters(self, df_clustered, centers, min_trips):
    for idx, center in centers.iterrows():
        cluster_points = df_clustered[df_clustered['cluster'] == idx]
        trip_count = len(cluster_points)

        if trip_count < min_trips:
            continue

        unique_cars = cluster_points['car_id'].nunique()
        avg_distance = cluster_points['drive_dist'].mean()
        total_distance = cluster_points['drive_dist'].sum()

        # 중요도 점수 계산
        importance = (
            trip_count * 0.4 +
            unique_cars * 0.3 +
            total_distance * 0.3
        )

        summaries.append({
            'cluster_id': idx,
            'trip_count': trip_count,
            'unique_cars': unique_cars,
            'avg_distance': avg_distance,
            'total_distance': total_distance,
            'center_lat': center['lat'],
            'center_lng': center['lon'],
            'importance_score': importance
        })
```

### 중요도 점수 공식
```
importance_score = trip_count × 0.4 + unique_cars × 0.3 + total_distance × 0.3
```

| 요소 | 가중치 | 설명 |
|------|--------|------|
| `trip_count` | 0.4 | 해당 지역 운행 건수 |
| `unique_cars` | 0.3 | 고유 차량 수 |
| `total_distance` | 0.3 | 총 운행 거리 |

### 서비스 부족 지역 분석
**위치**: [region_clustering.py:229-238](../src/services/region_clustering.py#L229-L238)

```python
def _find_underserved_areas(self, demand_centers, car_locations, threshold_km):
    underserved = []
    for lat, lon in demand_centers:
        if not car_locations:
            underserved.append({'lat': lat, 'lon': lon})
            continue

        min_dist = min(
            self._haversine_km(lat, lon, clat, clon)
            for clat, clon in car_locations
        )

        if min_dist > threshold_km:
            underserved.append({
                'lat': lat,
                'lon': lon,
                'min_distance_km': min_dist
            })
```

### 커버리지 분석
**위치**: [region_clustering.py:240-254](../src/services/region_clustering.py#L240-L254)

```python
def _compute_current_coverage(self, demand_centers, car_locations, threshold_km):
    covered = 0
    for lat, lon in demand_centers:
        distances = [self._haversine_km(lat, lon, clat, clon)
                     for clat, clon in car_locations]
        nearest = min(distances)
        is_covered = nearest <= threshold_km
        covered += 1 if is_covered else 0

    return {
        'covered_ratio': covered / len(demand_centers),
        'centers': details
    }
```

## 거리 계산

### Haversine 공식
**위치**: [region_clustering.py:214-227](../src/services/region_clustering.py#L214-L227)

두 GPS 좌표 간의 거리를 km 단위로 계산합니다.

```python
def _haversine_km(self, lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # 지구 반지름 (km)
    p1 = np.radians([lat1, lon1])
    p2 = np.radians([lat2, lon2])
    d = p2 - p1
    a = np.sin(d[0]/2)**2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(d[1]/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return float(R * c)
```

## 시각화 상세

### 1. 클러스터 맵
**위치**: [region_clustering.py:256-272](../src/services/region_clustering.py#L256-L272)

```python
def _plot_clusters(self, df_clustered, centers) -> str:
    fig, ax = plt.subplots(figsize=(8, 10))

    # 클러스터 포인트 (색상별 구분)
    valid_points = df_clustered[df_clustered['cluster'] >= 0]
    ax.scatter(valid_points['lon'], valid_points['lat'],
               c=valid_points['cluster'], cmap='tab10', s=20, alpha=0.6)

    # 클러스터 중심점 (X 마커)
    size = centers['count'] * 5 if 'count' in centers.columns else 50
    ax.scatter(centers['lon'], centers['lat'],
               c='black', s=size, marker='X', label='센터')

    ax.set_title('지역 수요 클러스터')
    ax.set_xlabel('경도')
    ax.set_ylabel('위도')
```

### 2. 추천 위치 맵
**위치**: [region_clustering.py:274-289](../src/services/region_clustering.py#L274-L289)

```python
def _plot_recommendations(self, centers, underserved) -> str:
    # 기존 수요 중심 (회색)
    ax.scatter(centers['lon'], centers['lat'],
               c='gray', s=size, marker='o', alpha=0.3, label='수요 중심')

    # 서비스 부족 지역 (빨간 X)
    if underserved:
        ulon = [u['lon'] for u in underserved]
        ulat = [u['lat'] for u in underserved]
        ax.scatter(ulon, ulat, c='red', s=80, marker='X',
                   label='추천 위치(부족 지역)')

    ax.set_title('추천 위치 (서비스 부족 지역)')
```

## 차량 위치 데이터

### 현재 차량 위치 조회
**위치**: [region_clustering.py:185-212](../src/services/region_clustering.py#L185-L212)

```python
def _load_car_locations(self) -> List[Tuple[float, float]]:
    query = """
    SELECT last_latitude AS lat, last_longitude AS lon
    FROM car
    WHERE last_latitude IS NOT NULL AND last_longitude IS NOT NULL
      AND last_latitude != 0 AND last_longitude != 0
    """

    df = get_data_from_db(query)

    # 좌표 유효성 검증
    df = df[np.isfinite(df['lat']) & np.isfinite(df['lon'])]

    return list(df[['lat', 'lon']].itertuples(index=False, name=None))
```

## 사용 예시

### Python 코드에서 직접 사용

```python
from src.services.region_clustering import RegionClusteringAnalyzer

analyzer = RegionClusteringAnalyzer()

# K-means 클러스터링
result = analyzer.analyze(
    k=7,
    use_end_points=True,
    threshold_km=3.0
)

if result['success']:
    # 중요도 순 클러스터
    for cluster in result['importance_ranking'][:3]:
        print(f"클러스터 {cluster['cluster_id']}: "
              f"운행 {cluster['trip_count']}건, "
              f"중심 ({cluster['center_lat']:.4f}, {cluster['center_lng']:.4f})")

    # 서비스 부족 지역
    for area in result['underserved_areas']:
        print(f"추천 위치: ({area['lat']:.4f}, {area['lon']:.4f}) "
              f"- 최근접 차량 {area['min_distance_km']:.1f}km")

    # 커버리지
    coverage = result['current_coverage']
    print(f"현재 커버리지: {coverage['covered_ratio']:.1%}")
```

### DBSCAN 사용 예시

```python
# DBSCAN으로 밀도 기반 클러스터링
result = analyzer.analyze(
    clustering_method='dbscan',
    eps_km=1.5,      # 1.5km 반경 내 포인트 그룹화
    min_trips=10     # 최소 10건 이상 운행 필요
)

# 노이즈 제외된 클러스터만 분석됨
```

## K-means vs DBSCAN 선택 가이드

| 상황 | 추천 알고리즘 |
|------|---------------|
| 클러스터 수를 알고 있음 | K-means |
| 클러스터 수를 모름 | DBSCAN |
| 구형 클러스터 예상 | K-means |
| 불규칙 형태 클러스터 | DBSCAN |
| 노이즈/이상치가 많음 | DBSCAN |
| 빠른 성능 필요 | K-means |

## 의존성

| 패키지 | 용도 |
|--------|------|
| pandas | 데이터 처리 |
| numpy | 수치 연산, 좌표 검증 |
| sklearn.cluster | KMeans, DBSCAN |
| matplotlib | 시각화 |
| seaborn | 색상 팔레트 |

## 제한사항

1. **K-means 클러스터 수**: 1-50개
2. **좌표 유효성**: 0,0 좌표 및 NaN 자동 제외
3. **DBSCAN eps**: km를 위도/경도로 변환 (근사값)
4. **캐싱**: 30분간 동일 파라미터 결과 캐싱

---

**관련 문서**: [[Module-Daily-Forecast]] | [[Module-Preference-Analysis]] | [[API-Reference]]
