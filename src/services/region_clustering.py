"""
지역 클러스터링 분석 모듈 (서비스 레이어)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from typing import Dict, Any, Tuple, List
import base64
import io
import logging

plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)
from ..utils.cache import cache_result


class RegionClusteringAnalyzer:
    def __init__(self):
        self.default_k = 5

    @cache_result(duration=1800)
    def analyze(self, start_date: str = None, end_date: str = None,
                k: int = 5, use_end_points: bool = True,
                method: str = 'kmeans', min_trips: int = 5,
                eps_km: float = 1.0, threshold_km: float = 5.0) -> Dict[str, Any]:
        try:
            k = int(k)
            if method.lower() == 'kmeans' and (k < 1 or k > 50):
                return {"success": False, "message": "k는 1~50 사이여야 합니다.", "visualizations": {}}

            df = self._load_data(start_date, end_date, use_end_points)
            if df.empty:
                return {"success": False, "message": "클러스터링할 위치 데이터가 없습니다.", "visualizations": {}}

            coords = df[['lat', 'lon']].dropna()
            # Additional validation for numeric coordinates
            coords = coords[(coords['lat'] != 0) & (coords['lon'] != 0)]  # Remove 0,0 coordinates
            # Convert to numeric and check finite values safely
            coords = coords[pd.to_numeric(coords['lat'], errors='coerce').notna()]
            coords = coords[pd.to_numeric(coords['lon'], errors='coerce').notna()]
            coords['lat'] = pd.to_numeric(coords['lat'], errors='coerce')
            coords['lon'] = pd.to_numeric(coords['lon'], errors='coerce')
            coords = coords[np.isfinite(coords['lat'].astype(float)) & np.isfinite(coords['lon'].astype(float))]
            
            if method.lower() == 'kmeans' and len(coords) < k:
                return {"success": False, "message": f"데이터 수({len(coords)})가 k({k})보다 적습니다.", "visualizations": {}}

            # 클러스터링
            if method.lower() == 'dbscan':
                eps_deg = eps_km / 111.0
                model = DBSCAN(eps=eps_deg, min_samples=max(3, min_trips))
                labels = model.fit_predict(coords.values)
                df_clustered = df.copy()
                df_clustered['cluster'] = labels
                centers = self._compute_dbscan_centers(df_clustered)
            else:
                model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = model.fit_predict(coords.values)
                df_clustered = df.copy()
                df_clustered['cluster'] = labels
                centers = pd.DataFrame(model.cluster_centers_, columns=['lat', 'lon'])
                cluster_sizes = df_clustered[df_clustered['cluster'] >= 0]['cluster'].value_counts().sort_index()
                centers['count'] = cluster_sizes.values if len(cluster_sizes) == len(centers) else 0

            # 요약/중요도
            cluster_summary, importance_ranking = self._summarize_clusters(df_clustered, centers, min_trips)

            # 커버리지/부족 지역
            car_locations = self._load_car_locations()
            underserved = self._find_underserved_areas(centers[['lat','lon']].values.tolist(), car_locations, threshold_km)
            coverage = self._compute_current_coverage(centers[['lat','lon']].values.tolist(), car_locations, threshold_km)

            charts = {}
            try:
                charts['cluster_map'] = self._plot_clusters(df_clustered, centers)
            except Exception as e:
                logger.error(f"클러스터 차트 오류: {e}")
                charts['cluster_map'] = ""
            try:
                charts['recommendation_map'] = self._plot_recommendations(centers, underserved)
            except Exception as e:
                logger.error(f"추천 차트 오류: {e}")
                charts['recommendation_map'] = ""

            return {
                "success": True,
                "message": "지역 클러스터링이 완료되었습니다.",
                "visualizations": charts,
                "cluster_summary": cluster_summary,
                "importance_ranking": importance_ranking,
                "recommended_locations": underserved,
                "underserved_areas": underserved,
                "current_coverage": coverage
            }
        except Exception as e:
            logger.error(f"클러스터링 분석 오류: {str(e)}")
            return {"success": False, "message": "서버 내부 오류가 발생했습니다.", "visualizations": {}}

    def _load_data(self, start_date: str, end_date: str, use_end_points: bool) -> pd.DataFrame:
        from ..data_loader import get_data_from_db

        where_clauses = ["dl.start_time IS NOT NULL"]
        if start_date:
            where_clauses.append(f"DATE(dl.start_time) >= '{start_date}'")
        if end_date:
            where_clauses.append(f"DATE(dl.start_time) <= '{end_date}'")
        where_sql = " AND ".join(where_clauses)

        if use_end_points:
            query = f"""
            SELECT 
                dl.end_latitude AS lat, dl.end_longitude AS lon,
                dl.start_latitude, dl.start_longitude,
                dl.start_point, dl.end_point,
                dl.drive_dist, dl.car_id
            FROM drive_log dl
            WHERE {where_sql} AND dl.end_latitude IS NOT NULL AND dl.end_longitude IS NOT NULL
            """
        else:
            query = f"""
            SELECT 
                dl.start_latitude AS lat, dl.start_longitude AS lon,
                dl.start_latitude, dl.start_longitude,
                dl.start_point, dl.end_point,
                dl.drive_dist, dl.car_id
            FROM drive_log dl
            WHERE {where_sql} AND dl.start_latitude IS NOT NULL AND dl.start_longitude IS NOT NULL
            """

        df = get_data_from_db(query)
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Ensure coordinate columns are numeric
        for col in ['lat', 'lon']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _compute_dbscan_centers(self, df_clustered: pd.DataFrame) -> pd.DataFrame:
        valid = df_clustered[df_clustered['cluster'] >= 0]
        centers = valid.groupby('cluster')[['lat','lon']].mean().reset_index(drop=True)
        counts = valid['cluster'].value_counts().sort_index()
        centers['count'] = counts.values
        return centers

    def _summarize_clusters(self, df_clustered: pd.DataFrame, centers: pd.DataFrame, min_trips: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        summaries = []
        for idx, center in centers.iterrows():
            cluster_points = df_clustered[df_clustered.get('cluster', -1) == idx]
            trip_count = len(cluster_points)
            if trip_count < min_trips:
                continue
            unique_cars = int(cluster_points['car_id'].nunique()) if 'car_id' in cluster_points.columns else None
            avg_distance = float(cluster_points['drive_dist'].mean()) if 'drive_dist' in cluster_points.columns else None
            total_distance = float(cluster_points['drive_dist'].sum()) if 'drive_dist' in cluster_points.columns else None
            importance = (
                (trip_count or 0) * 0.4 +
                (unique_cars or 0) * 0.3 +
                (total_distance or 0.0) * 0.3
            )
            summaries.append({
                'cluster_id': int(idx),
                'trip_count': int(trip_count),
                'unique_cars': unique_cars,
                'avg_distance': avg_distance,
                'total_distance': total_distance,
                'center_lat': float(center['lat']),
                'center_lng': float(center['lon']),
                'importance_score': float(importance)
            })

        ranking = sorted(summaries, key=lambda x: x['importance_score'], reverse=True)
        return summaries, ranking

    def _load_car_locations(self) -> List[Tuple[float, float]]:
        from ..data_loader import get_data_from_db
        query = """
        SELECT last_latitude AS lat, last_longitude AS lon
        FROM car
        WHERE last_latitude IS NOT NULL AND last_longitude IS NOT NULL
          AND last_latitude != 0 AND last_longitude != 0
        """
        df = get_data_from_db(query)
        if df is None or df.empty:
            return []
        
        # Additional validation for numeric coordinates
        df = df.dropna()
        # Convert to numeric and safely check for finite values
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df = df.dropna()  # Remove rows where conversion failed
        df = df[np.isfinite(df['lat'].astype(float)) & np.isfinite(df['lon'].astype(float))]
        
        # Convert to float to ensure proper type
        try:
            df['lat'] = df['lat'].astype(float)
            df['lon'] = df['lon'].astype(float)
        except (ValueError, TypeError):
            return []
            
        return list(df[['lat','lon']].itertuples(index=False, name=None))

    def _haversine_km(self, lat1, lon1, lat2, lon2) -> float:
        # Convert to float and handle None/invalid values
        try:
            lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
        except (ValueError, TypeError):
            return float('inf')  # Return large distance for invalid coordinates
            
        R = 6371.0
        p1 = np.radians([lat1, lon1])
        p2 = np.radians([lat2, lon2])
        d = p2 - p1
        a = np.sin(d[0]/2)**2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(d[1]/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return float(R * c)

    def _find_underserved_areas(self, demand_centers: List[Tuple[float, float]], car_locations: List[Tuple[float, float]], threshold_km: float) -> List[Dict[str, float]]:
        underserved = []
        for lat, lon in demand_centers:
            if not car_locations:
                underserved.append({'lat': float(lat), 'lon': float(lon)})
                continue
            min_dist = min(self._haversine_km(lat, lon, clat, clon) for clat, clon in car_locations)
            if min_dist > threshold_km:
                underserved.append({'lat': float(lat), 'lon': float(lon), 'min_distance_km': float(min_dist)})
        return underserved

    def _compute_current_coverage(self, demand_centers: List[Tuple[float, float]], car_locations: List[Tuple[float, float]], threshold_km: float) -> Dict[str, Any]:
        if not demand_centers:
            return {'covered_ratio': 0.0, 'centers': []}
        covered = 0
        details = []
        for lat, lon in demand_centers:
            if not car_locations:
                details.append({'lat': float(lat), 'lon': float(lon), 'nearest_km': None, 'covered': False})
                continue
            distances = [self._haversine_km(lat, lon, clat, clon) for clat, clon in car_locations]
            nearest = min(distances)
            is_covered = nearest <= threshold_km
            covered += 1 if is_covered else 0
            details.append({'lat': float(lat), 'lon': float(lon), 'nearest_km': float(nearest), 'covered': is_covered})
        return {'covered_ratio': float(covered / len(demand_centers)), 'centers': details}

    def _plot_clusters(self, df_clustered: pd.DataFrame, centers: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(8, 10))

        valid_points = df_clustered[df_clustered.get('cluster', -1) >= 0]
        ax.scatter(valid_points['lon'], valid_points['lat'], c=valid_points['cluster'],
                   cmap='tab10', s=20, alpha=0.6)
        size = centers['count'] * 5 if 'count' in centers.columns else 50
        ax.scatter(centers['lon'], centers['lat'], c='black', s=size, marker='X', label='센터')

        ax.set_title('지역 수요 클러스터')
        ax.set_xlabel('경도')
        ax.set_ylabel('위도')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_recommendations(self, centers: pd.DataFrame, underserved: List[Dict[str, float]]) -> str:
        fig, ax = plt.subplots(figsize=(8, 10))
        if len(centers) > 0:
            size = centers['count'] * 5 if 'count' in centers.columns else 50
            ax.scatter(centers['lon'], centers['lat'], c='gray', s=size, marker='o', alpha=0.3, label='수요 중심')
        if underserved:
            ulon = [u['lon'] for u in underserved]
            ulat = [u['lat'] for u in underserved]
            ax.scatter(ulon, ulat, c='red', s=80, marker='X', label='추천 위치(부족 지역)')
        ax.set_title('추천 위치 (서비스 부족 지역)')
        ax.set_xlabel('경도')
        ax.set_ylabel('위도')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='jpeg', dpi=75, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/jpeg;base64,{image_base64}"


def create_region_clustering_api():
    from flask import request, jsonify

    analyzer = RegionClusteringAnalyzer()

    def region_clustering():
        try:
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            k = request.args.get('k', None)
            if k is None:
                k = request.args.get('n_clusters', 5)
            method = request.args.get('clustering_method', 'kmeans')
            min_trips = int(request.args.get('min_trips', 5))
            use_end_points = request.args.get('use_end_points', 'true').lower() == 'true'
            eps_km = float(request.args.get('eps_km', 1.0))
            threshold_km = float(request.args.get('threshold_km', 5.0))

            result = analyzer.analyze(start_date, end_date, k, use_end_points, method, min_trips, eps_km, threshold_km)
            return jsonify(result), 200 if result.get('success') else 400
        except Exception as e:
            logger.error(f"지역 클러스터링 API 오류: {str(e)}")
            return jsonify({"success": False, "message": "서버 내부 오류가 발생했습니다.", "visualizations": {}}), 500

    return region_clustering


