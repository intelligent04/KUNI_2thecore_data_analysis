"""
일별 운행 대수 예측 모듈
주어진 기간의 일별 운행량을 집계하고, SARIMA 모델로 향후 7~30일을 예측
그래프(base64)와 함께 반환
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Tuple
import base64
import io
import logging
import warnings
from datetime import timedelta
from itertools import product

# 폰트 설정은 utils.font_config에서 처리
from ..utils.font_config import setup_korean_font
setup_korean_font()

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

            # 예측 (SARIMA 모델, 타겟: unique_cars)
            forecast_df, model_metrics = self._forecast_sarima(daily_stats, forecast_days)

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

    def _forecast_sarima(self, daily_stats: pd.DataFrame, forecast_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # 시계열 데이터 준비
        full_dates = pd.date_range(daily_stats['date'].min(), daily_stats['date'].max(), freq='D')
        series = daily_stats.set_index(pd.to_datetime(daily_stats['date']))['unique_cars'].reindex(full_dates, fill_value=0)

        # 시계열 전처리
        series_processed, transformation_info = self._preprocess_timeseries(series)

        if len(series_processed) < 14:  # 최소 2주 데이터 필요
            # 데이터가 부족할 경우 단순 평균 예측
            mean_value = series_processed.mean()
            forecast_dates = [series.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'date': [d.date() for d in forecast_dates],
                'predicted_unique_cars': [float(mean_value)] * forecast_days
            })
            metrics = {
                'method': 'mean_fallback',
                'mae': float(np.mean(np.abs(series_processed - mean_value))),
                'aic': None,
                'bic': None
            }
            return forecast_df, metrics

        # SARIMA 모델 최적 파라미터 찾기
        best_params, best_seasonal_params = self._find_best_sarima_params(series_processed)

        try:
            # SARIMA 모델 학습
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = SARIMAX(series_processed,
                               order=best_params,
                               seasonal_order=best_seasonal_params,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                fitted_model = model.fit(disp=False)

            # 예측 수행
            forecast = fitted_model.forecast(steps=forecast_days)
            forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()

            # 음수 값 보정
            forecast = np.maximum(forecast, 0)

            # 예측 결과 DataFrame 생성
            forecast_dates = [series.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'date': [d.date() for d in forecast_dates],
                'predicted_unique_cars': forecast.values.astype(float),
                'lower_ci': np.maximum(forecast_ci.iloc[:, 0].values, 0).astype(float),
                'upper_ci': forecast_ci.iloc[:, 1].values.astype(float)
            })

            # 모델 평가 지표
            fitted_values = fitted_model.fittedvalues
            metrics = {
                'method': 'SARIMA',
                'sarima_order': best_params,
                'seasonal_order': best_seasonal_params,
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'mae': float(mean_absolute_error(series_processed, fitted_values)),
                'ljung_box_pvalue': float(acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)['lb_pvalue'].iloc[-1])
            }

            return forecast_df, metrics

        except Exception as e:
            logger.warning(f"SARIMA 모델 학습 실패: {e}, 단순 ARIMA로 대체")
            return self._fallback_arima_forecast(series_processed, forecast_days)

    def _preprocess_timeseries(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """시계열 전처리: 정상성 확인 및 변환"""
        original_series = series.copy()
        transformation_info = {'transformations': []}

        # 영점 처리 (log 변환을 위해 작은 값 추가)
        if (series <= 0).any():
            series = series + 1
            transformation_info['transformations'].append('add_constant')

        # 정상성 검정 (ADF test)
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05
        transformation_info['initial_stationarity'] = is_stationary

        # 차분 필요 여부 확인
        if not is_stationary:
            # 1차 차분
            series_diff = series.diff().dropna()
            if len(series_diff) > 0:
                adf_diff = adfuller(series_diff)
                if adf_diff[1] < 0.05:
                    transformation_info['transformations'].append('first_difference')
                    transformation_info['final_stationarity'] = True
                else:
                    transformation_info['final_stationarity'] = False

        return series, transformation_info

    def _find_best_sarima_params(self, series: pd.Series) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """SARIMA 최적 파라미터 탐색 (AIC 기준)"""
        # 데이터 길이에 따른 파라미터 범위 조정
        max_p = min(3, len(series) // 10)
        max_d = min(2, len(series) // 20)
        max_q = min(3, len(series) // 10)

        # 계절성 주기 (일별 데이터에서 주간 패턴 고려)
        seasonal_period = 7 if len(series) >= 21 else 0

        best_aic = np.inf
        best_params = (1, 1, 1)
        best_seasonal_params = (0, 0, 0, 0)

        # 비계절 파라미터 탐색
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)

        for p, d, q in product(p_values, d_values, q_values):
            if p + d + q == 0:
                continue

            try:
                # 계절성 파라미터 (간단하게 고정)
                if seasonal_period > 0 and len(series) >= seasonal_period * 3:
                    seasonal_params = (1, 0, 1, seasonal_period)
                else:
                    seasonal_params = (0, 0, 0, 0)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_params,
                                   enforce_stationarity=False, enforce_invertibility=False)
                    fitted = model.fit(disp=False, maxiter=100)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_params = (p, d, q)
                        best_seasonal_params = seasonal_params

            except Exception:
                continue

        return best_params, best_seasonal_params

    def _fallback_arima_forecast(self, series: pd.Series, forecast_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """SARIMA 실패시 단순 ARIMA 대체 모델"""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = ARIMA(series, order=(1, 1, 1))
                fitted_model = model.fit()

            forecast = fitted_model.forecast(steps=forecast_days)
            forecast = np.maximum(forecast, 0)

            forecast_dates = [series.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'date': [d.date() for d in forecast_dates],
                'predicted_unique_cars': forecast.astype(float)
            })

            metrics = {
                'method': 'ARIMA_fallback',
                'aic': float(fitted_model.aic),
                'mae': float(mean_absolute_error(series, fitted_model.fittedvalues))
            }

            return forecast_df, metrics

        except Exception:
            # 최종 대체: 단순 평균
            mean_value = series.mean()
            forecast_dates = [series.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'date': [d.date() for d in forecast_dates],
                'predicted_unique_cars': [float(mean_value)] * forecast_days
            })
            metrics = {
                'method': 'mean_fallback',
                'mae': float(np.mean(np.abs(series - mean_value)))
            }
            return forecast_df, metrics

    def _plot_usage_with_prediction(self, daily_stats: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(14, 8))

        # 실제 데이터 플롯
        actual_dates = pd.to_datetime(daily_stats['date'])
        ax.plot(actual_dates, daily_stats['unique_cars'],
                label='실제 운행 차량 수', color='#1f77b4', linewidth=2, marker='o', markersize=4)

        # 예측 데이터 플롯
        forecast_dates = pd.to_datetime(forecast_df['date'])
        ax.plot(forecast_dates, forecast_df['predicted_unique_cars'],
                label='SARIMA 예측', color='#ff7f0e', linestyle='--', linewidth=2, marker='s', markersize=4)

        # 신뢰구간 표시 (있는 경우)
        if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
            ax.fill_between(forecast_dates,
                           forecast_df['lower_ci'],
                           forecast_df['upper_ci'],
                           color='#ff7f0e', alpha=0.2, label='95% 신뢰구간')

        # 예측 시작점 표시
        if len(daily_stats) > 0 and len(forecast_df) > 0:
            last_actual = daily_stats.iloc[-1]
            first_forecast = forecast_df.iloc[0]
            ax.plot([actual_dates.iloc[-1], forecast_dates.iloc[0]],
                   [last_actual['unique_cars'], first_forecast['predicted_unique_cars']],
                   color='gray', linestyle=':', alpha=0.7)

        ax.set_title('일별 운행 차량 수: 실측값과 SARIMA 예측', fontsize=16, fontweight='bold')
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('운행 차량 수', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 축 범위 조정
        ax.set_ylim(bottom=0)

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


