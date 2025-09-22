"""
간소화된 연도별 트렌드 분석 모듈
sklearn을 활용한 효율적인 트렌드 분석 및 시각화
"""

import pandas as pd
import numpy as np
# matplotlib/seaborn은 지연 로딩 (_get_mpl) 사용

def _get_mpl():
    # 폰트 설정 적용
    from .utils.font_config import get_matplotlib
    return get_matplotlib()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, List, Any, Optional
import base64
import io
import logging
from datetime import datetime

# 한글 폰트 설정은 _get_mpl 내부에서 지연 설정

logger = logging.getLogger(__name__)


class SimpleTrendAnalyzer:
    """간소화된 트렌드 분석 클래스"""
    
    def __init__(self):
        self.brand_colors = {'현대': '#1f77b4', '기아': '#ff7f0e', '제네시스': '#2ca02c'}
        # 동적 색상 배정용 팔레트 및 캐시 (지연 확장)
        self.dynamic_colors = {}
        self.color_palette = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e']
        self.palette_idx = 0
        self.current_year = datetime.now().year
        
    def analyze_yearly_trend(self, start_year: int = 2020, end_year: int = 2025, 
                           top_n: int = 5) -> Dict[str, Any]:
        """메인 트렌드 분석 함수"""
        try:
            # 데이터 로드
            df = self._load_trend_data(start_year, end_year)
            if df.empty:
                return {"success": False, "message": "분석할 데이터가 없습니다.", "visualizations": {}}
            
            # 트렌드 분석 수행
            trend_results = self._analyze_trends(df, top_n)
            
            # 시각화 생성
            visualizations = self._create_trend_charts(df, trend_results, top_n)
            
            return {
                "success": True,
                "message": f"{start_year}-{end_year} 연도별 트렌드 분석이 완료되었습니다.",
                "visualizations": visualizations
            }
            
        except Exception as e:
            logger.error(f"트렌드 분석 중 오류: {str(e)}")
            return {
                "success": False,
                "message": f"트렌드 분석 중 오류가 발생했습니다: {str(e)}",
                "visualizations": {}
            }
    
    def _load_trend_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """트렌드 분석용 데이터 로드"""
        from .data_loader import get_data_from_db
        
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
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 기본 전처리
        df['car_age'] = df['drive_year'] - df['car_year']
        
        # 연도별 총 운행 건수 계산
        yearly_totals = df.groupby('drive_year')['drive_count'].sum()
        df['market_share'] = df['drive_count'] / df['drive_year'].map(yearly_totals)
        
        return df
    
    def _analyze_trends(self, df: pd.DataFrame, top_n: int) -> Dict[str, Any]:
        """트렌드 분석 수행"""
        results = {}
        
        # 1. 브랜드별 트렌드 분석
        results['brand_trends'] = self._analyze_brand_trends(df)
        
        # 2. 상위 모델 트렌드 분석
        results['model_trends'] = self._analyze_model_trends(df, top_n)
        
        # 3. 차량 연식별 선호도 분석
        results['car_age_trends'] = self._analyze_car_age_trends(df)
        
        # 4. 시장 점유율 변화
        results['market_share_evolution'] = self._analyze_market_share_evolution(df)
        
        return results
    
    def _analyze_brand_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """브랜드별 트렌드 분석"""
        brand_trends = {}
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand].groupby('drive_year')['market_share'].sum().reset_index()
            
            if len(brand_data) >= 2:
                # sklearn 선형회귀로 트렌드 계산
                X = brand_data['drive_year'].values.reshape(-1, 1)
                y = brand_data['market_share'].values
                
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
                
                brand_trends[brand] = {
                    'slope': float(slope),
                    'r2_score': float(r2),
                    'trend_direction': trend_direction,
                    'yearly_data': brand_data.set_index('drive_year')['market_share'].to_dict()
                }
        
        return brand_trends
    
    def _analyze_model_trends(self, df: pd.DataFrame, top_n: int) -> Dict[str, Any]:
        """상위 모델별 트렌드 분석"""
        # 전체 기간 상위 모델 선별
        model_totals = df.groupby(['brand', 'model'])['drive_count'].sum().sort_values(ascending=False)
        top_models = model_totals.head(top_n).index.tolist()
        
        model_trends = {}
        
        for brand, model in top_models:
            model_data = df[(df['brand'] == brand) & (df['model'] == model)]
            yearly_data = model_data.groupby('drive_year')['market_share'].sum().reset_index()
            
            if len(yearly_data) >= 2:
                X = yearly_data['drive_year'].values.reshape(-1, 1)
                y = yearly_data['market_share'].values
                
                model_reg = LinearRegression()
                model_reg.fit(X, y)
                
                slope = model_reg.coef_[0]
                r2 = r2_score(y, model_reg.predict(X))
                
                model_key = f"{brand}_{model}"
                model_trends[model_key] = {
                    'brand': brand,
                    'model': model,
                    'slope': float(slope),
                    'r2_score': float(r2),
                    'trend_direction': 'increasing' if slope > 0.005 else 'decreasing' if slope < -0.005 else 'stable',
                    'yearly_data': yearly_data.set_index('drive_year')['market_share'].to_dict()
                }
        
        return model_trends
    
    def _analyze_car_age_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """차량 연식별 선호도 트렌드"""
        # 연령대별 분류
        df_age = df.copy()
        df_age['age_category'] = pd.cut(df_age['car_age'], 
                                       bins=[-float('inf'), 2, 5, 10, float('inf')],
                                       labels=['신차(0-2년)', '준신차(3-5년)', '중고차(6-10년)', '노후차(11년+)'])
        
        age_trends = {}
        
        for age_cat in df_age['age_category'].unique():
            if pd.isna(age_cat):
                continue
                
            age_data = df_age[df_age['age_category'] == age_cat]
            yearly_shares = age_data.groupby('drive_year')['market_share'].sum().reset_index()
            
            if len(yearly_shares) >= 2:
                X = yearly_shares['drive_year'].values.reshape(-1, 1)
                y = yearly_shares['market_share'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                slope = model.coef_[0]
                
                age_trends[str(age_cat)] = {
                    'slope': float(slope),
                    'trend_direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                    'yearly_data': yearly_shares.set_index('drive_year')['market_share'].to_dict()
                }
        
        return age_trends
    
    def _analyze_market_share_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시장 점유율 진화 분석"""
        yearly_brand_share = df.groupby(['drive_year', 'brand'])['market_share'].sum().unstack(fill_value=0)
        
        evolution = {
            'yearly_shares': yearly_brand_share.to_dict(),
            'volatility': {},  # 브랜드별 변동성
            'growth_rate': {}  # 브랜드별 성장률
        }
        
        # 각 브랜드별 변동성과 성장률 계산
        for brand in yearly_brand_share.columns:
            shares = yearly_brand_share[brand].values
            
            # 변동성 (변동계수)
            volatility = np.std(shares) / np.mean(shares) if np.mean(shares) > 0 else 0
            evolution['volatility'][brand] = float(volatility)
            
            # 성장률 (첫해 대비 마지막해)
            if len(shares) >= 2 and shares[0] > 0:
                growth_rate = (shares[-1] - shares[0]) / shares[0]
                evolution['growth_rate'][brand] = float(growth_rate)
        
        return evolution
    
    def _create_trend_charts(self, df: pd.DataFrame, trend_results: Dict[str, Any], 
                           top_n: int) -> Dict[str, str]:
        """트렌드 차트 생성"""
        charts = {}
        
        try:
            charts['brand_trend_lines'] = self._create_brand_trend_chart(trend_results['brand_trends'])
        except Exception as e:
            logger.error(f"브랜드 트렌드 차트 오류: {e}")
            charts['brand_trend_lines'] = ""
        
        try:
            charts['model_ranking_change'] = self._create_model_ranking_chart(trend_results['model_trends'])
        except Exception as e:
            logger.error(f"모델 랭킹 차트 오류: {e}")
            charts['model_ranking_change'] = ""
        
        try:
            charts['car_age_preference'] = self._create_car_age_chart(trend_results['car_age_trends'])
        except Exception as e:
            logger.error(f"차량 연식 차트 오류: {e}")
            charts['car_age_preference'] = ""
        
        try:
            charts['market_share_evolution'] = self._create_market_evolution_chart(trend_results['market_share_evolution'])
        except Exception as e:
            logger.error(f"시장점유율 차트 오류: {e}")
            charts['market_share_evolution'] = ""
        
        try:
            charts['trend_summary'] = self._create_trend_summary_chart(trend_results['brand_trends'])
        except Exception as e:
            logger.error(f"트렌드 요약 차트 오류: {e}")
            charts['trend_summary'] = ""
        
        return charts
    
    def _create_brand_trend_chart(self, brand_trends: Dict[str, Any]) -> str:
        """브랜드별 트렌드 라인 차트"""
        plt, _ = _get_mpl()
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for brand, data in brand_trends.items():
            yearly_data = data['yearly_data']
            years = list(yearly_data.keys())
            shares = list(yearly_data.values())
            
            color = self._color_for_brand(brand)
            ax.plot(years, shares, marker='o', linewidth=2.5, markersize=8,
                   label=f"{brand} ({data['trend_direction']})", color=color)
            
            # 트렌드 라인 추가
            if len(years) >= 2:
                X = np.array(years).reshape(-1, 1)
                model = LinearRegression().fit(X, shares)
                trend_line = model.predict(X)
                ax.plot(years, trend_line, linestyle='--', alpha=0.7, color=color)
        
        ax.set_title('브랜드별 시장 점유율 트렌드', fontsize=16, fontweight='bold')
        ax.set_xlabel('연도', fontsize=12)
        ax.set_ylabel('시장 점유율', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_model_ranking_chart(self, model_trends: Dict[str, Any]) -> str:
        """모델 랭킹 변화 차트"""
        plt, _ = _get_mpl()
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 상위 5개 모델만 표시
        top_5_models = list(model_trends.items())[:5]
        
        y_positions = range(len(top_5_models))
        
        for i, (model_key, data) in enumerate(top_5_models):
            brand = data['brand']
            model = data['model']
            trend_direction = data['trend_direction']
            slope = data['slope']
            
            # 트렌드 방향에 따른 색상
            if trend_direction == 'increasing':
                color = 'green'
                symbol = '↗'
            elif trend_direction == 'decreasing':
                color = 'red'
                symbol = '↘'
            else:
                color = 'gray'
                symbol = '→'
            
            ax.barh(i, abs(slope) * 100, color=color, alpha=0.7)
            ax.text(abs(slope) * 100 + 0.01, i, f"{model} {symbol}", 
                   va='center', fontweight='bold')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{data['brand']}" for _, data in top_5_models])
        ax.set_xlabel('트렌드 강도 (%)')
        ax.set_title('상위 모델별 트렌드 변화', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_car_age_chart(self, car_age_trends: Dict[str, Any]) -> str:
        """차량 연식별 선호도 차트"""
        plt, _ = _get_mpl()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        age_categories = list(car_age_trends.keys())
        slopes = [data['slope'] for data in car_age_trends.values()]
        
        colors = ['green' if slope > 0 else 'red' if slope < 0 else 'gray' for slope in slopes]
        
        bars = ax.bar(age_categories, slopes, color=colors, alpha=0.7)
        
        # 값 표시
        for bar, slope in zip(bars, slopes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),
                   f'{slope:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('차량 연식별 선호도 트렌드', fontsize=16, fontweight='bold')
        ax.set_ylabel('트렌드 기울기')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_market_evolution_chart(self, market_evolution: Dict[str, Any]) -> str:
        """시장 점유율 진화 차트"""
        plt, _ = _get_mpl()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 변동성 차트
        volatility = market_evolution['volatility']
        brands = list(volatility.keys())
        vol_values = list(volatility.values())
        colors = [self._color_for_brand(b) for b in brands]
        
        ax1.bar(brands, vol_values, color=colors, alpha=0.7)
        ax1.set_title('브랜드별 시장점유율 변동성')
        ax1.set_ylabel('변동성 (변동계수)')
        
        # 2. 성장률 차트
        growth_rate = market_evolution['growth_rate']
        brands_growth = list(growth_rate.keys())
        growth_values = list(growth_rate.values())
        colors_growth = ['green' if g > 0 else 'red' if g < 0 else 'gray' for g in growth_values]
        
        bars = ax2.bar(brands_growth, growth_values, color=colors_growth, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('브랜드별 점유율 성장률')
        ax2.set_ylabel('성장률')
        
        # 성장률 값 표시
        for bar, growth in zip(bars, growth_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                    f'{growth:.2%}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _color_for_brand(self, brand: str) -> str:
        if brand in self.brand_colors:
            return self.brand_colors[brand]
        if brand in self.dynamic_colors:
            return self.dynamic_colors[brand]
        # 팔레트 고갈 시 seaborn 팔레트로 확장
        if self.palette_idx >= len(self.color_palette):
            try:
                plt, _sns = _get_mpl()
                extra = _sns.color_palette('tab20', n_colors=20).as_hex()
                for c in extra:
                    if c not in self.color_palette:
                        self.color_palette.append(c)
            except Exception:
                pass
        color = self.color_palette[self.palette_idx % len(self.color_palette)]
        self.palette_idx += 1
        self.dynamic_colors[brand] = color
        return color
    
    def _create_trend_summary_chart(self, brand_trends: Dict[str, Any]) -> str:
        """트렌드 요약 차트"""
        plt, _ = _get_mpl()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        brands = list(brand_trends.keys())
        trend_directions = [data['trend_direction'] for data in brand_trends.values()]
        slopes = [data['slope'] for data in brand_trends.values()]
        r2_scores = [data['r2_score'] for data in brand_trends.values()]
        
        # 트렌드 방향별 색상
        direction_colors = {
            'increasing': 'green',
            'decreasing': 'red', 
            'stable': 'gray'
        }
        
        colors = [direction_colors.get(direction, 'gray') for direction in trend_directions]
        
        # 기울기를 바 높이로, R² 점수를 알파값으로 사용
        bars = ax.bar(brands, slopes, color=colors, alpha=0.7)
        
        # 트렌드 방향 표시
        for i, (brand, direction, slope, r2) in enumerate(zip(brands, trend_directions, slopes, r2_scores)):
            symbol = '↗' if direction == 'increasing' else '↘' if direction == 'decreasing' else '→'
            ax.text(i, slope + 0.001 if slope >= 0 else slope - 0.003, 
                   f'{symbol} R²={r2:.2f}', ha='center', 
                   va='bottom' if slope >= 0 else 'top', fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('브랜드별 트렌드 요약', fontsize=16, fontweight='bold')
        ax.set_ylabel('트렌드 기울기')
        ax.set_xlabel('브랜드')
        
        # 범례
        legend_elements = [plt.Rectangle((0,0),1,1, color='green', alpha=0.7, label='증가 추세'),
                          plt.Rectangle((0,0),1,1, color='red', alpha=0.7, label='감소 추세'),
                          plt.Rectangle((0,0),1,1, color='gray', alpha=0.7, label='안정 추세')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Figure를 base64로 변환"""
        import matplotlib.pyplot as plt
        buffer = io.BytesIO()
        fig.savefig(buffer, format='jpeg', dpi=75, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/jpeg;base64,{image_base64}"


# Flask API 엔드포인트
def create_simple_trend_api():
    """간소화된 트렌드 분석 API 생성"""
    from flask import request, jsonify
    
    analyzer = SimpleTrendAnalyzer()
    
    def yearly_trend():
        """API 엔드포인트"""
        try:
            start_year = int(request.args.get('start_year', 2020))
            end_year = int(request.args.get('end_year', 2025))
            top_n = int(request.args.get('top_n', 5))
            
            # 파라미터 검증
            if start_year >= end_year:
                return jsonify({
                    "success": False,
                    "message": "시작 연도는 종료 연도보다 작아야 합니다.",
                    "visualizations": {}
                }), 400
            
            if top_n < 1 or top_n > 20:
                return jsonify({
                    "success": False,
                    "message": "top_n은 1-20 사이의 값이어야 합니다.",
                    "visualizations": {}
                }), 400
            
            # 분석 실행
            result = analyzer.analyze_yearly_trend(start_year, end_year, top_n)
            
            return jsonify(result), 200 if result['success'] else 400
            
        except ValueError as e:
            return jsonify({
                "success": False,
                "message": "파라미터는 숫자여야 합니다.",
                "visualizations": {}
            }), 400
        except Exception as e:
            logger.error(f"트렌드 분석 API 오류: {str(e)}")
            return jsonify({
                "success": False,
                "message": "서버 내부 오류가 발생했습니다.",
                "visualizations": {}
            }), 500
    
    return yearly_trend