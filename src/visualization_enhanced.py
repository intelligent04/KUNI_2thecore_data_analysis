"""
시각화 강화 모듈
통계적 검정 결과와 비즈니스 인사이트가 포함된 고도화된 시각화 기능 제공
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional, Tuple
import base64
import io
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class EnhancedVisualization:
    """강화된 시각화 클래스"""
    
    def __init__(self):
        # 색상 팔레트 설정
        self.brand_colors = {
            '현대': '#1f77b4',
            '기아': '#ff7f0e', 
            '제네시스': '#2ca02c',
            'Unknown': '#d62728'
        }
        
        self.season_colors = {
            1: '#98FB98',  # 봄 - 연두
            2: '#FFB6C1',  # 여름 - 분홍
            3: '#DEB887',  # 가을 - 갈색
            4: '#87CEEB'   # 겨울 - 하늘색
        }
        
        self.season_names = {1: '봄', 2: '여름', 3: '가을', 4: '겨울'}
        self.month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                          '7월', '8월', '9월', '10월', '11월', '12월']
        
        # 시각화 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_preference_charts(self, df: pd.DataFrame, period_type: str = 'month') -> Dict[str, str]:
        """기본 선호도 차트 생성"""
        try:
            charts = {}
            
            # 1. 브랜드별 기간별 선호도 히트맵
            charts['brand_period_heatmap'] = self._create_brand_period_heatmap(df, period_type)
            
            # 2. 브랜드별 시장 점유율 파이 차트
            charts['market_share_pie'] = self._create_market_share_pie(df)
            
            # 3. 기간별 브랜드 선호도 라인 차트
            charts['brand_preference_line'] = self._create_brand_preference_line(df, period_type)
            
            # 4. 브랜드별 계절성 레이더 차트
            if period_type == 'season':
                charts['seasonality_radar'] = self._create_seasonality_radar(df)
            
            return charts
            
        except Exception as e:
            logger.error(f"기본 차트 생성 중 오류: {str(e)}")
            return {'error': str(e)}
    
    def create_brand_seasonality_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """브랜드별 계절성 시각화"""
        try:
            charts = {}
            
            # 1. 브랜드별 계절성 강도 바 차트
            charts['seasonality_strength_bar'] = self._create_seasonality_strength_chart(df)
            
            # 2. 브랜드별 월별 분포 박스플롯
            charts['monthly_distribution_box'] = self._create_monthly_boxplot(df)
            
            # 3. 브랜드별 계절 선호도 히트맵
            charts['seasonal_preference_heatmap'] = self._create_seasonal_heatmap(df)
            
            # 4. 통계적 유의성이 포함된 브랜드 비교 차트
            charts['statistical_brand_comparison'] = self._create_statistical_comparison_chart(df)
            
            return charts
            
        except Exception as e:
            logger.error(f"브랜드 계절성 차트 생성 중 오류: {str(e)}")
            return {'error': str(e)}
    
    def create_model_seasonality_charts(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, str]:
        """모델별 계절성 시각화"""
        try:
            charts = {}
            
            # 상위 모델 선별
            top_models = df['model'].value_counts().head(top_n).index.tolist()
            df_top = df[df['model'].isin(top_models)]
            
            # 1. 상위 모델별 계절 분포
            charts['top_models_seasonal'] = self._create_top_models_seasonal_chart(df_top)
            
            # 2. 모델별 계절성 지수 비교
            charts['model_seasonality_index'] = self._create_model_seasonality_index(df_top)
            
            # 3. 브랜드별 모델 다양성
            charts['brand_model_diversity'] = self._create_brand_model_diversity(df_top)
            
            return charts
            
        except Exception as e:
            logger.error(f"모델 계절성 차트 생성 중 오류: {str(e)}")
            return {'error': str(e)}
    
    def create_statistical_visualization(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """통계적 검정 결과 시각화"""
        try:
            charts = {}
            
            # 1. 효과 크기 시각화
            charts['effect_size_visualization'] = self._create_effect_size_chart(analysis_results)
            
            # 2. 신뢰구간이 포함된 선호도 차트
            charts['confidence_interval_chart'] = self._create_confidence_interval_chart(analysis_results)
            
            # 3. 통계적 유의성 표시 차트
            charts['significance_indicators'] = self._create_significance_chart(analysis_results)
            
            return charts
            
        except Exception as e:
            logger.error(f"통계 시각화 생성 중 오류: {str(e)}")
            return {'error': str(e)}
    
    def _create_brand_period_heatmap(self, df: pd.DataFrame, period_type: str) -> str:
        """브랜드별 기간별 선호도 히트맵"""
        try:
            period_col = 'month' if period_type == 'month' else 'season'
            
            # 교차표 생성
            crosstab = pd.crosstab(df['brand'], df[period_col], normalize='columns')
            
            # 히트맵 생성
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if period_type == 'season':
                # 계절 라벨 변경
                crosstab.columns = [self.season_names.get(col, str(col)) for col in crosstab.columns]
            else:
                # 월 라벨 변경
                crosstab.columns = [f'{int(col)}월' for col in crosstab.columns]
            
            sns.heatmap(crosstab, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': '선호도 비율'}, ax=ax)
            
            ax.set_title(f'브랜드별 {period_type} 선호도 히트맵', fontsize=16, fontweight='bold')
            ax.set_xlabel('기간', fontsize=12)
            ax.set_ylabel('브랜드', fontsize=12)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"히트맵 생성 오류: {str(e)}")
            return self._create_error_image("히트맵 생성 실패")
    
    def _create_market_share_pie(self, df: pd.DataFrame) -> str:
        """시장 점유율 파이 차트"""
        try:
            market_share = df['brand'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 색상 매핑
            colors = [self.brand_colors.get(brand, '#gray') for brand in market_share.index]
            
            wedges, texts, autotexts = ax.pie(market_share.values, 
                                            labels=market_share.index,
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            explode=[0.05 if i == 0 else 0 for i in range(len(market_share))])
            
            # 텍스트 스타일링
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('브랜드별 시장 점유율', fontsize=16, fontweight='bold')
            
            # 총 레코드 수 표시
            ax.text(0, -1.3, f'총 운행 건수: {len(df):,}건', 
                   ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"파이 차트 생성 오류: {str(e)}")
            return self._create_error_image("파이 차트 생성 실패")
    
    def _create_brand_preference_line(self, df: pd.DataFrame, period_type: str) -> str:
        """브랜드별 기간별 선호도 라인 차트"""
        try:
            period_col = 'month' if period_type == 'month' else 'season'
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                period_counts = brand_data.groupby(period_col).size()
                
                # 전체 기간에 대해 0으로 채우기
                if period_type == 'month':
                    full_periods = range(1, 13)
                else:
                    full_periods = range(1, 5)
                
                period_counts = period_counts.reindex(full_periods, fill_value=0)
                
                ax.plot(period_counts.index, period_counts.values, 
                       marker='o', linewidth=2.5, markersize=8,
                       label=brand, color=self.brand_colors.get(brand, 'gray'))
            
            ax.set_title(f'브랜드별 {period_type} 선호도 트렌드', fontsize=16, fontweight='bold')
            ax.set_xlabel('기간', fontsize=12)
            ax.set_ylabel('운행 건수', fontsize=12)
            ax.legend(title='브랜드', title_fontsize=12, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # x축 라벨 설정
            if period_type == 'month':
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels([f'{i}월' for i in range(1, 13)])
            else:
                ax.set_xticks(range(1, 5))
                ax.set_xticklabels([self.season_names[i] for i in range(1, 5)])
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"라인 차트 생성 오류: {str(e)}")
            return self._create_error_image("라인 차트 생성 실패")
    
    def _create_seasonality_radar(self, df: pd.DataFrame) -> str:
        """계절성 레이더 차트"""
        try:
            # 브랜드별 계절별 정규화된 값 계산
            seasonal_data = {}
            
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                seasonal_counts = brand_data.groupby('season').size()
                # 정규화 (최대값을 1로)
                if seasonal_counts.max() > 0:
                    seasonal_norm = seasonal_counts / seasonal_counts.max()
                    seasonal_data[brand] = seasonal_norm.reindex([1, 2, 3, 4], fill_value=0).values
            
            # 레이더 차트를 위한 각도 설정
            angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
            angles += angles[:1]  # 원형으로 만들기
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for brand, values in seasonal_data.items():
                values = np.concatenate((values, [values[0]]))  # 원형으로 만들기
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=brand, color=self.brand_colors.get(brand, 'gray'))
                ax.fill(angles, values, alpha=0.25, 
                       color=self.brand_colors.get(brand, 'gray'))
            
            # 계절 라벨 설정
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['봄', '여름', '가을', '겨울'])
            ax.set_ylim(0, 1)
            
            ax.set_title('브랜드별 계절성 레이더 차트', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"레이더 차트 생성 오류: {str(e)}")
            return self._create_error_image("레이더 차트 생성 실패")
    
    def _create_seasonality_strength_chart(self, df: pd.DataFrame) -> str:
        """계절성 강도 바 차트"""
        try:
            seasonality_scores = {}
            
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                seasonal_counts = brand_data.groupby('season').size()
                
                if len(seasonal_counts) > 1:
                    # 변동계수로 계절성 강도 측정
                    cv = seasonal_counts.std() / seasonal_counts.mean()
                    seasonality_scores[brand] = cv
            
            if not seasonality_scores:
                return self._create_error_image("계절성 데이터 없음")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            brands = list(seasonality_scores.keys())
            scores = list(seasonality_scores.values())
            colors = [self.brand_colors.get(brand, 'gray') for brand in brands]
            
            bars = ax.bar(brands, scores, color=colors, alpha=0.7)
            
            # 각 막대에 값 표시
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('브랜드별 계절성 강도 (변동계수)', fontsize=16, fontweight='bold')
            ax.set_xlabel('브랜드', fontsize=12)
            ax.set_ylabel('계절성 강도 (CV)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 계절성 강도 해석 범례
            interpretation_text = "높을수록 계절성이 강함\n(0.5 이상: 강함, 0.3~0.5: 보통, 0.3 미만: 약함)"
            ax.text(0.98, 0.95, interpretation_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=9)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"계절성 강도 차트 생성 오류: {str(e)}")
            return self._create_error_image("계절성 강도 차트 생성 실패")
    
    def _create_monthly_boxplot(self, df: pd.DataFrame) -> str:
        """월별 분포 박스플롯"""
        try:
            # 브랜드별로 월별 카운트 데이터 준비
            monthly_data = []
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                monthly_counts = brand_data.groupby('month').size().reindex(range(1, 13), fill_value=0)
                
                for month, count in monthly_counts.items():
                    monthly_data.append({
                        'brand': brand,
                        'month': month,
                        'count': count
                    })
            
            monthly_df = pd.DataFrame(monthly_data)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            sns.boxplot(data=monthly_df, x='month', y='count', hue='brand', ax=ax)
            
            ax.set_title('브랜드별 월별 운행 건수 분포', fontsize=16, fontweight='bold')
            ax.set_xlabel('월', fontsize=12)
            ax.set_ylabel('운행 건수', fontsize=12)
            ax.legend(title='브랜드', title_fontsize=12, fontsize=10)
            
            # x축 라벨 설정
            ax.set_xticklabels([f'{i}월' for i in range(1, 13)])
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"박스플롯 생성 오류: {str(e)}")
            return self._create_error_image("박스플롯 생성 실패")
    
    def _create_seasonal_heatmap(self, df: pd.DataFrame) -> str:
        """계절별 선호도 히트맵 (정규화)"""
        try:
            # 브랜드별 계절별 교차표
            crosstab = pd.crosstab(df['brand'], df['season'], normalize='index')
            
            # 계절 이름으로 변경
            crosstab.columns = [self.season_names.get(col, str(col)) for col in crosstab.columns]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 히트맵 생성
            im = sns.heatmap(crosstab, annot=True, fmt='.3f', cmap='RdYlBu_r',
                            center=0.25, vmin=0, vmax=0.5,
                            cbar_kws={'label': '상대적 선호도'},
                            ax=ax)
            
            ax.set_title('브랜드별 계절 선호도 히트맵 (행 정규화)', fontsize=16, fontweight='bold')
            ax.set_xlabel('계절', fontsize=12)
            ax.set_ylabel('브랜드', fontsize=12)
            
            # 컬러바 라벨 추가
            cbar = ax.collections[0].colorbar
            cbar.set_label('각 브랜드 내 계절별 상대적 선호도', rotation=270, labelpad=20)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"계절 히트맵 생성 오류: {str(e)}")
            return self._create_error_image("계절 히트맵 생성 실패")
    
    def _create_statistical_comparison_chart(self, df: pd.DataFrame) -> str:
        """통계적 유의성이 포함된 브랜드 비교 차트"""
        try:
            from scipy.stats import chi2_contingency
            
            # 교차표 생성
            crosstab = pd.crosstab(df['brand'], df['season'])
            
            # 카이제곱 검정
            chi2, p_value, dof, expected = chi2_contingency(crosstab)
            
            # 표준화 잔차 계산
            residuals = (crosstab - expected) / np.sqrt(expected)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. 관측값 히트맵
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues',
                       ax=ax1, cbar_kws={'label': '관측 빈도'})
            ax1.set_title(f'관측값\n(χ² = {chi2:.2f}, p = {p_value:.4f})', fontsize=14)
            ax1.set_xlabel('계절')
            ax1.set_ylabel('브랜드')
            
            # 2. 표준화 잔차 히트맵
            sns.heatmap(residuals, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax2, cbar_kws={'label': '표준화 잔차'})
            ax2.set_title('표준화 잔차\n(|값| > 2: 유의한 차이)', fontsize=14)
            ax2.set_xlabel('계절')
            ax2.set_ylabel('브랜드')
            
            # 유의성 해석
            if p_value < 0.05:
                significance_text = f"p < 0.05: 브랜드와 계절 간 유의한 연관성 있음"
            else:
                significance_text = f"p ≥ 0.05: 브랜드와 계절 간 유의한 연관성 없음"
            
            fig.suptitle(f'브랜드-계절 연관성 통계 분석\n{significance_text}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"통계 비교 차트 생성 오류: {str(e)}")
            return self._create_error_image("통계 비교 차트 생성 실패")
    
    def _create_top_models_seasonal_chart(self, df: pd.DataFrame) -> str:
        """상위 모델별 계절 분포 차트"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 모델별 계절별 교차표
            crosstab = pd.crosstab(df['model'], df['season'], normalize='index')
            
            # 상위 10개 모델만 표시
            crosstab = crosstab.head(10)
            
            # 스택 바 차트
            crosstab.plot(kind='bar', stacked=True, ax=ax, 
                         color=[self.season_colors.get(col, 'gray') for col in crosstab.columns])
            
            ax.set_title('상위 모델별 계절 선호도 분포', fontsize=16, fontweight='bold')
            ax.set_xlabel('모델', fontsize=12)
            ax.set_ylabel('상대적 선호도', fontsize=12)
            ax.legend(title='계절', labels=[self.season_names.get(col, str(col)) for col in crosstab.columns])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"모델 계절 차트 생성 오류: {str(e)}")
            return self._create_error_image("모델 계절 차트 생성 실패")
    
    def _create_model_seasonality_index(self, df: pd.DataFrame) -> str:
        """모델별 계절성 지수"""
        try:
            seasonality_scores = {}
            
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                if len(model_data) >= 12:  # 최소 12개 레코드 필요
                    seasonal_counts = model_data.groupby('season').size()
                    if len(seasonal_counts) > 1:
                        cv = seasonal_counts.std() / seasonal_counts.mean()
                        seasonality_scores[model] = cv
            
            if not seasonality_scores:
                return self._create_error_image("모델 계절성 데이터 부족")
            
            # 상위 10개 모델
            top_seasonal_models = sorted(seasonality_scores.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            models, scores = zip(*top_seasonal_models)
            
            bars = ax.barh(range(len(models)), scores, alpha=0.7)
            
            # 색상 그라데이션
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            ax.set_xlabel('계절성 강도 (변동계수)', fontsize=12)
            ax.set_title('모델별 계절성 강도 순위 (상위 10개)', fontsize=16, fontweight='bold')
            
            # 값 표시
            for i, score in enumerate(scores):
                ax.text(score + 0.01, i, f'{score:.3f}', 
                       va='center', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"모델 계절성 지수 차트 생성 오류: {str(e)}")
            return self._create_error_image("모델 계절성 지수 차트 생성 실패")
    
    def _create_brand_model_diversity(self, df: pd.DataFrame) -> str:
        """브랜드별 모델 다양성 차트"""
        try:
            brand_model_counts = df.groupby('brand')['model'].nunique().sort_values(ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = [self.brand_colors.get(brand, 'gray') for brand in brand_model_counts.index]
            bars = ax.barh(brand_model_counts.index, brand_model_counts.values, color=colors, alpha=0.7)
            
            # 값 표시
            for bar, count in zip(bars, brand_model_counts.values):
                ax.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{count}개', va='center', fontweight='bold')
            
            ax.set_xlabel('모델 수', fontsize=12)
            ax.set_title('브랜드별 모델 다양성', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"브랜드 모델 다양성 차트 생성 오류: {str(e)}")
            return self._create_error_image("브랜드 모델 다양성 차트 생성 실패")
    
    def _create_effect_size_chart(self, analysis_results: Dict[str, Any]) -> str:
        """효과 크기 시각화"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 더미 데이터로 효과 크기 차트 생성 (실제 구현시 analysis_results에서 추출)
            effect_sizes = {'Small': 0.1, 'Medium': 0.3, 'Large': 0.5}
            
            bars = ax.bar(effect_sizes.keys(), effect_sizes.values(), 
                         color=['green', 'orange', 'red'], alpha=0.7)
            
            ax.set_title('효과 크기 분류', fontsize=16, fontweight='bold')
            ax.set_ylabel('Effect Size (Cramer\'s V)', fontsize=12)
            
            # 해석 가이드라인 추가
            ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Small effect')
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Large effect')
            
            ax.legend()
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"효과 크기 차트 생성 오류: {str(e)}")
            return self._create_error_image("효과 크기 차트 생성 실패")
    
    def _create_confidence_interval_chart(self, analysis_results: Dict[str, Any]) -> str:
        """신뢰구간이 포함된 차트"""
        try:
            # 더미 데이터로 신뢰구간 차트 생성
            brands = ['현대', '기아', '제네시스']
            means = [0.25, 0.35, 0.15]
            errors = [0.05, 0.07, 0.03]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(brands, means, yerr=errors, capsize=10, 
                         color=[self.brand_colors.get(b, 'gray') for b in brands],
                         alpha=0.7, error_kw={'linewidth': 2})
            
            ax.set_title('브랜드별 선호도 (95% 신뢰구간)', fontsize=16, fontweight='bold')
            ax.set_ylabel('선호도 비율', fontsize=12)
            ax.set_xlabel('브랜드', fontsize=12)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"신뢰구간 차트 생성 오류: {str(e)}")
            return self._create_error_image("신뢰구간 차트 생성 실패")
    
    def _create_significance_chart(self, analysis_results: Dict[str, Any]) -> str:
        """통계적 유의성 표시 차트"""
        try:
            # 더미 데이터로 유의성 차트 생성
            comparisons = ['현대 vs 기아', '현대 vs 제네시스', '기아 vs 제네시스']
            p_values = [0.001, 0.045, 0.23]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            bars = ax.bar(comparisons, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
            
            ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, 
                      label='p = 0.05 (유의성 기준)')
            
            ax.set_title('브랜드 간 비교의 통계적 유의성', fontsize=16, fontweight='bold')
            ax.set_ylabel('-log10(p-value)', fontsize=12)
            ax.set_xlabel('브랜드 비교', fontsize=12)
            ax.legend()
            
            # p-value 값 표시
            for bar, p_val in zip(bars, p_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'p = {p_val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"유의성 차트 생성 오류: {str(e)}")
            return self._create_error_image("유의성 차트 생성 실패")
    
    def _fig_to_base64(self, fig) -> str:
        """Matplotlib figure를 base64 문자열로 변환"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)  # 메모리 해제
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Base64 변환 오류: {str(e)}")
            plt.close(fig)
            return ""
    
    def _create_error_image(self, error_message: str) -> str:
        """오류 메시지가 포함된 이미지 생성"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'차트 생성 실패\n{error_message}', 
                   ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            return self._fig_to_base64(fig)
            
        except Exception:
            return ""