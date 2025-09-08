"""
일별 운행 대수 예측 모듈
주어진 기간의 일별 운행량을 집계하고, 다항 회귀로 향후 7~30일을 예측
그래프(base64)와 함께 반환
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Dict, Any
import base64
import io
import logging
from datetime import timedelta

plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)
from ..utils.cache import cache_result


class DailyForecastAnalyzer:
    def __init__(self):
        self.max_forecast_days = 30

    @cache_result(duration=1800)
    def analyze(self, start_date: str = None, end_date: str = None,
                forecast_days: int = 7) -> Dict[str, Any]:
        try:
            forecast_days = int(forecast_days)
            if forecast_days < 1 or forecast_days > self.max_forecast_days:
                return {"success": False, "message": f"forecast_days는 1~{self.max_forecast_days} 사이여야 합니다.", "visualizations": {}}

            df = self._load_data(start_date, end_date)
            if df.empty:
                return {"success": False, "message": "분석할 데이터가 없습니다.", "visualizations": {}}

            # 일별 통계 집계
            daily_stats = df.groupby('date').agg(
                unique_cars=('car_id', 'nunique'),
                total_trips=('drive_log_id', 'count'),
                total_distance=('drive_dist', 'sum')
            ).reset_index().sort_values('date')

            # 요일 패턴
            tmp = daily_stats.copy()
            tmp['weekday'] = pd.to_datetime(tmp['date']).dt.dayofweek
            weekday_pattern = tmp.groupby('weekday')['unique_cars'].mean().to_dict()

            # 예측 (다항 회귀 degree=2, 타겟: unique_cars)
            forecast_df, model_metrics = self._forecast_polynomial(daily_stats, forecast_days)

            charts = {}
            try:
                charts['usage_trend_with_prediction'] = self._plot_usage_with_prediction(daily_stats, forecast_df)
            except Exception as e:
                logger.error(f"차트 생성 오류(usage_trend_with_prediction): {e}")
                charts['usage_trend_with_prediction'] = ""
            try:
                charts['weekday_pattern'] = self._plot_weekday_pattern(weekday_pattern)
            except Exception as e:
                logger.error(f"차트 생성 오류(weekday_pattern): {e}")
                charts['weekday_pattern'] = ""

            # JSON 직렬화를 위한 날짜 변환
            daily_stats_json = daily_stats.copy()
            daily_stats_json['date'] = daily_stats_json['date'].astype(str)
            forecast_json = forecast_df.copy()
            forecast_json['date'] = forecast_json['date'].astype(str)

            return {
                "success": True,
                "message": "일별 운행량 예측이 완료되었습니다.",
                "visualizations": charts,
                "historical_data": daily_stats_json.to_dict(orient='records'),
                "predictions": forecast_json.to_dict(orient='records'),
                "weekday_patterns": weekday_pattern,
                "model_accuracy": model_metrics
            }
        except Exception as e:
            logger.error(f"예측 분석 오류: {str(e)}")
            return {"success": False, "message": "서버 내부 오류가 발생했습니다.", "visualizations": {}}

    def _load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        from ..data_loader import get_data_from_db

        where_clauses = ["dl.start_time IS NOT NULL"]
        if start_date:
            where_clauses.append(f"DATE(dl.start_time) >= '{start_date}'")
        if end_date:
            where_clauses.append(f"DATE(dl.start_time) <= '{end_date}'")

        where_sql = " AND ".join(where_clauses)

        query = f"""
        SELECT 
            dl.start_time,
            dl.car_id,
            dl.drive_log_id,
            dl.drive_dist
        FROM drive_log dl
        WHERE {where_sql}
        """

        df = get_data_from_db(query)
        if df is None or df.empty:
            return pd.DataFrame()

        df['start_time'] = pd.to_datetime(df['start_time'])
        df['date'] = df['start_time'].dt.date
        return df[['date', 'car_id', 'drive_log_id', 'drive_dist']]

    def _forecast_polynomial(self, daily_stats: pd.DataFrame, forecast_days: int):
        # 결측 날짜 0 채우기 (unique_cars 기준)
        full_dates = pd.date_range(daily_stats['date'].min(), daily_stats['date'].max(), freq='D')
        series = daily_stats.set_index(pd.to_datetime(daily_stats['date']))['unique_cars'].reindex(full_dates, fill_value=0)

        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        # in-sample accuracy
        y_hat = model.predict(X_poly)
        metrics = {
            'r2': float(r2_score(y, y_hat)),
            'mae': float(mean_absolute_error(y, y_hat))
        }

        future_index = np.arange(len(series), len(series) + forecast_days).reshape(-1, 1)
        future_poly = poly.transform(future_index)
        y_pred = np.clip(model.predict(future_poly), a_min=0, a_max=None)

        forecast_dates = [series.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
        forecast_df = pd.DataFrame({
            'date': [d.date() for d in forecast_dates],
            'predicted_unique_cars': y_pred.astype(float)
        })
        return forecast_df, metrics

    def _plot_usage_with_prediction(self, daily_stats: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(pd.to_datetime(daily_stats['date']), daily_stats['unique_cars'], label='실제(차량 수)', color='#1f77b4')
        ax.plot(pd.to_datetime(forecast_df['date']), forecast_df['predicted_unique_cars'], label='예측(차량 수)', color='#ff7f0e', linestyle='--')

        ax.set_title('일별 운행 차량 수: 실측과 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('운행 차량 수')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_weekday_pattern(self, weekday_pattern: Dict[int, float]) -> str:
        fig, ax = plt.subplots(figsize=(8, 4))
        weekdays = list(sorted(weekday_pattern.keys()))
        values = [weekday_pattern[w] for w in weekdays]
        labels = ['월','화','수','목','금','토','일']
        ax.bar([labels[w] for w in weekdays], values, color='#2ca02c', alpha=0.8)
        ax.set_title('요일별 평균 운행 차량 수')
        ax.set_ylabel('평균 차량 수')
        ax.grid(True, axis='y', alpha=0.2)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='jpeg', dpi=75, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/jpeg;base64,{image_base64}"


def create_daily_forecast_api():
    from flask import request, jsonify

    analyzer = DailyForecastAnalyzer()

    def daily_forecast():
        try:
            start_date = request.args.get('start_date')  # YYYY-MM-DD
            end_date = request.args.get('end_date')      # YYYY-MM-DD
            # 설계도 호환: predict_days 지원, 우선순위: forecast_days > predict_days
            forecast_days = request.args.get('forecast_days', None)
            if forecast_days is None:
                forecast_days = request.args.get('predict_days', 7)

            result = analyzer.analyze(start_date, end_date, forecast_days)
            return jsonify(result), 200 if result.get('success') else 400
        except Exception as e:
            logger.error(f"일별 예측 API 오류: {str(e)}")
            return jsonify({"success": False, "message": "서버 내부 오류가 발생했습니다.", "visualizations": {}}), 500

    return daily_forecast


