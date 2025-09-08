"""
개선된 선호도 분석 API 모듈
데이터 품질 검증, 고도화된 계절성 분석, 통계적 유의성 검정을 통합한 종합적 API
"""

from flask import request, jsonify, current_app
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import traceback

# 내부 모듈 import
from .data_loader import get_data_from_db
from .data_quality import DataQualityValidator, validate_input_parameters
from .seasonality_analysis import SeasonalityAnalyzer
from .statistical_tests import StatisticalTestSuite
from .visualization_enhanced import EnhancedVisualization

logger = logging.getLogger(__name__)


class PreferenceAnalysisAPI:
    """개선된 선호도 분석 API 클래스"""
    
    def __init__(self):
        self.validator = DataQualityValidator()
        self.seasonality_analyzer = SeasonalityAnalyzer()
        self.statistical_suite = StatisticalTestSuite()
        self.visualizer = EnhancedVisualization()
        
        # 분석 설정
        self.analysis_config = {
            'min_records_for_analysis': 30,
            'min_brands_for_comparison': 2,
            'min_periods_for_seasonality': 6,
            'significance_level': 0.05,
            'top_models_to_analyze': 10
        }
    
    def analyze_preference_by_period(self, year: Optional[str] = None, 
                                   period_type: str = 'month') -> Dict[str, Any]:
        """
        개선된 월별/계절별 선호도 분석 API
        
        Args:
            year: 분석할 연도 (선택적)
            period_type: 분석 기간 타입 ('month' 또는 'season')
            
        Returns:
            종합적 선호도 분석 결과
        """
        try:
            # 1. 입력 파라미터 검증
            param_validation = validate_input_parameters(year, period_type)
            if not param_validation['valid']:
                return {
                    'success': False,
                    'errors': param_validation['errors'],
                    'warnings': param_validation['warnings']
                }
            
            # 정제된 파라미터 사용
            clean_params = param_validation['cleaned_params']
            
            # 2. 데이터 로드
            data_load_result = self._load_analysis_data(clean_params['year'])
            if not data_load_result['success']:
                return data_load_result
            
            df = data_load_result['data']
            
            # 3. 데이터 품질 검증
            quality_report = self.validator.validate_data_quality(df)
            
            # 품질 점수가 너무 낮으면 경고와 함께 제한적 분석 수행
            if quality_report['quality_score'] < 0.6:
                logger.warning(f"데이터 품질 점수가 낮음: {quality_report['quality_score']:.2f}")
            
            # 4. 선호도 분석 수행
            preference_results = self._perform_preference_analysis(
                df, clean_params['period_type'], quality_report
            )
            
            # 5. 통계적 유의성 검정
            statistical_results = self._perform_statistical_analysis(
                df, clean_params['period_type']
            )
            
            # 6. 시각화 생성
            visualization_results = self._generate_visualizations(
                df, clean_params['period_type'], preference_results
            )
            
            # 7. 비즈니스 인사이트 생성
            business_insights = self._generate_business_insights(
                preference_results, statistical_results, quality_report
            )
            
            # 8. 종합 결과 반환
            return {
                'success': True,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_period': clean_params,
                    'total_records': len(df),
                    'analysis_config': self.analysis_config
                },
                'data_quality': quality_report,
                'preference_analysis': preference_results,
                'statistical_analysis': statistical_results,
                'visualizations': visualization_results,
                'business_insights': business_insights,
                'warnings': param_validation.get('warnings', [])
            }
            
        except Exception as e:
            logger.error(f"선호도 분석 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'분석 중 오류가 발생했습니다: {str(e)}',
                'error_type': type(e).__name__
            }
    
    def analyze_brand_seasonality_detailed(self, brands: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        브랜드별 상세 계절성 분석
        
        Args:
            brands: 분석할 브랜드 리스트 (None이면 전체)
            
        Returns:
            브랜드별 상세 계절성 분석 결과
        """
        try:
            # 데이터 로드
            data_load_result = self._load_analysis_data()
            if not data_load_result['success']:
                return data_load_result
            
            df = data_load_result['data']
            
            # 브랜드 필터링
            if brands:
                df = df[df['brand'].isin(brands)]
                if df.empty:
                    return {
                        'success': False,
                        'error': f'지정된 브랜드({brands})에 대한 데이터가 없습니다.'
                    }
            
            # 브랜드별 계절성 분석
            brand_seasonality = self.seasonality_analyzer.analyze_brand_seasonality(df)
            
            # 브랜드 비교 통계 검정
            brand_comparison = self.statistical_suite.brand_comparison_tests(df)
            
            # 브랜드별 시각화
            brand_visualizations = self.visualizer.create_brand_seasonality_charts(df)
            
            return {
                'success': True,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'brands_analyzed': df['brand'].unique().tolist(),
                    'total_records': len(df)
                },
                'brand_seasonality_analysis': brand_seasonality,
                'brand_comparison_tests': brand_comparison,
                'visualizations': brand_visualizations
            }
            
        except Exception as e:
            logger.error(f"브랜드 계절성 분석 중 오류 발생: {str(e)}")
            return {
                'success': False,
                'error': f'브랜드 계절성 분석 중 오류가 발생했습니다: {str(e)}'
            }
    
    def analyze_model_seasonality_detailed(self, top_n: int = 10) -> Dict[str, Any]:
        """
        모델별 상세 계절성 분석
        
        Args:
            top_n: 분석할 상위 모델 개수
            
        Returns:
            모델별 상세 계절성 분석 결과
        """
        try:
            # 데이터 로드
            data_load_result = self._load_analysis_data()
            if not data_load_result['success']:
                return data_load_result
            
            df = data_load_result['data']
            
            # 모델별 계절성 분석
            model_seasonality = self.seasonality_analyzer.analyze_model_seasonality(df, top_n)
            
            # 모델별 시각화
            model_visualizations = self.visualizer.create_model_seasonality_charts(df, top_n)
            
            return {
                'success': True,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'top_models_analyzed': top_n,
                    'total_records': len(df)
                },
                'model_seasonality_analysis': model_seasonality,
                'visualizations': model_visualizations
            }
            
        except Exception as e:
            logger.error(f"모델 계절성 분석 중 오류 발생: {str(e)}")
            return {
                'success': False,
                'error': f'모델 계절성 분석 중 오류가 발생했습니다: {str(e)}'
            }
    
    def _load_analysis_data(self, year: Optional[str] = None) -> Dict[str, Any]:
        """분석용 데이터 로드"""
        try:
            # 기본 쿼리 구성
            base_query = """
            SELECT 
                dl.start_time,
                dl.brand,
                dl.model,
                dl.car_id,
                c.car_year,
                c.car_type,
                dl.drive_dist,
                dl.start_latitude,
                dl.start_longitude,
                dl.end_latitude,
                dl.end_longitude,
                dl.memo
            FROM drive_log dl
            JOIN car c ON dl.car_id = c.car_id
            WHERE dl.start_time IS NOT NULL
            """
            
            # 연도 필터 추가
            if year:
                base_query += f" AND YEAR(dl.start_time) = {year}"
            
            base_query += " ORDER BY dl.start_time"
            
            # 데이터 로드
            df = get_data_from_db(base_query)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': '분석할 데이터가 없습니다.',
                    'data': pd.DataFrame()
                }
            
            # 기본 데이터 정제
            df = self._preprocess_data(df)
            
            return {
                'success': True,
                'data': df,
                'records_loaded': len(df)
            }
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return {
                'success': False,
                'error': f'데이터 로드 중 오류가 발생했습니다: {str(e)}',
                'data': pd.DataFrame()
            }
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            # 시간 데이터 처리
            df['start_time'] = pd.to_datetime(df['start_time'])
            
            # 시간 파생 변수 생성
            df['year'] = df['start_time'].dt.year
            df['month'] = df['start_time'].dt.month
            df['season'] = df['month'].map({
                12: 4, 1: 4, 2: 4,  # 겨울
                3: 1, 4: 1, 5: 1,   # 봄
                6: 2, 7: 2, 8: 2,   # 여름
                9: 3, 10: 3, 11: 3  # 가을
            })
            df['weekday'] = df['start_time'].dt.weekday
            df['hour'] = df['start_time'].dt.hour
            
            # 브랜드/모델 데이터 정제
            df['brand'] = df['brand'].fillna('Unknown').astype(str)
            df['model'] = df['model'].fillna('Unknown').astype(str)
            
            # 이상치 제거 (선택적)
            # 예: 너무 긴 거리는 제외
            if 'drive_dist' in df.columns:
                q99 = df['drive_dist'].quantile(0.99)
                df = df[df['drive_dist'] <= q99]
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            raise
    
    def _perform_preference_analysis(self, df: pd.DataFrame, period_type: str, 
                                   quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """선호도 분석 수행"""
        try:
            results = {}
            
            # 1. 기본 선호도 지표 계산
            preference_metrics = self.seasonality_analyzer.calculate_preference_metrics(
                df, period_type
            )
            results['preference_metrics'] = preference_metrics
            
            # 2. 브랜드별 계절성 분석
            brand_seasonality = self.seasonality_analyzer.analyze_brand_seasonality(df)
            results['brand_seasonality'] = brand_seasonality
            
            # 3. 상위 모델 분석
            top_models_analysis = self.seasonality_analyzer.analyze_model_seasonality(
                df, self.analysis_config['top_models_to_analyze']
            )
            results['top_models_analysis'] = top_models_analysis
            
            # 4. 시장 점유율 분석
            market_analysis = self._analyze_market_share(df, period_type)
            results['market_analysis'] = market_analysis
            
            # 5. 트렌드 분석 (시간이 충분할 경우)
            if len(df['year'].unique()) > 1:
                trend_analysis = self._analyze_temporal_trends(df, period_type)
                results['trend_analysis'] = trend_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"선호도 분석 수행 중 오류 발생: {str(e)}")
            raise
    
    def _perform_statistical_analysis(self, df: pd.DataFrame, period_type: str) -> Dict[str, Any]:
        """통계적 분석 수행"""
        try:
            results = {}
            
            # 1. 종합적 선호도 검정
            comprehensive_test = self.statistical_suite.comprehensive_preference_test(
                df, 'brand', 'season' if period_type == 'season' else 'month'
            )
            results['comprehensive_tests'] = comprehensive_test
            
            # 2. 계절성 유의성 검정
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                if len(brand_data) >= 24:  # 최소 2년치 월별 데이터
                    monthly_counts = brand_data.groupby('month').size()
                    seasonality_test = self.statistical_suite.seasonality_significance_test(
                        monthly_counts, 12
                    )
                    results[f'{brand}_seasonality_tests'] = seasonality_test
            
            # 3. 브랜드 간 비교 검정
            brand_comparison = self.statistical_suite.brand_comparison_tests(df)
            results['brand_comparison_tests'] = brand_comparison
            
            return results
            
        except Exception as e:
            logger.error(f"통계적 분석 수행 중 오류 발생: {str(e)}")
            raise
    
    def _generate_visualizations(self, df: pd.DataFrame, period_type: str, 
                               preference_results: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            visualizations = {}
            
            # 1. 기본 선호도 차트
            basic_charts = self.visualizer.create_preference_charts(df, period_type)
            visualizations.update(basic_charts)
            
            # 2. 브랜드별 계절성 차트
            brand_charts = self.visualizer.create_brand_seasonality_charts(df)
            visualizations.update(brand_charts)
            
            # 3. 모델별 분석 차트
            model_charts = self.visualizer.create_model_seasonality_charts(
                df, self.analysis_config['top_models_to_analyze']
            )
            visualizations.update(model_charts)
            
            # 4. 통계적 검정 시각화
            statistical_charts = self.visualizer.create_statistical_visualization(
                preference_results
            )
            visualizations.update(statistical_charts)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"시각화 생성 중 오류 발생: {str(e)}")
            # 시각화 실패시에도 분석은 계속 진행
            return {'visualization_error': str(e)}
    
    def _analyze_market_share(self, df: pd.DataFrame, period_type: str) -> Dict[str, Any]:
        """시장 점유율 분석"""
        try:
            period_col = 'month' if period_type == 'month' else 'season'
            
            # 전체 시장 점유율
            total_market_share = df['brand'].value_counts(normalize=True).to_dict()
            
            # 기간별 시장 점유율
            period_market_share = {}
            for period in df[period_col].unique():
                period_data = df[df[period_col] == period]
                period_share = period_data['brand'].value_counts(normalize=True).to_dict()
                period_market_share[str(period)] = period_share
            
            # 시장 점유율 변동성
            market_volatility = {}
            for brand in df['brand'].unique():
                brand_shares = []
                for period in df[period_col].unique():
                    period_data = df[df[period_col] == period]
                    brand_share = (period_data['brand'] == brand).mean()
                    brand_shares.append(brand_share)
                
                if brand_shares:
                    volatility = np.std(brand_shares) / np.mean(brand_shares) if np.mean(brand_shares) > 0 else 0
                    market_volatility[brand] = float(volatility)
            
            return {
                'total_market_share': total_market_share,
                'period_market_share': period_market_share,
                'market_volatility': market_volatility,
                'dominant_brand': max(total_market_share.items(), key=lambda x: x[1])[0],
                'market_concentration_hhi': sum(share**2 for share in total_market_share.values())
            }
            
        except Exception as e:
            logger.error(f"시장 점유율 분석 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_temporal_trends(self, df: pd.DataFrame, period_type: str) -> Dict[str, Any]:
        """시간적 트렌드 분석"""
        try:
            results = {}
            
            # 연도별 브랜드 점유율 변화
            yearly_trends = {}
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year]
                yearly_share = year_data['brand'].value_counts(normalize=True).to_dict()
                yearly_trends[str(year)] = yearly_share
            
            # 브랜드별 성장률 계산
            growth_rates = {}
            years = sorted(df['year'].unique())
            if len(years) >= 2:
                for brand in df['brand'].unique():
                    first_year_count = len(df[(df['year'] == years[0]) & (df['brand'] == brand)])
                    last_year_count = len(df[(df['year'] == years[-1]) & (df['brand'] == brand)])
                    
                    if first_year_count > 0:
                        growth_rate = (last_year_count - first_year_count) / first_year_count
                        growth_rates[brand] = float(growth_rate)
            
            results = {
                'yearly_trends': yearly_trends,
                'growth_rates': growth_rates,
                'analysis_period': {
                    'start_year': min(years) if years else None,
                    'end_year': max(years) if years else None,
                    'total_years': len(years)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"트렌드 분석 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _generate_business_insights(self, preference_results: Dict[str, Any], 
                                  statistical_results: Dict[str, Any], 
                                  quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """비즈니스 인사이트 생성"""
        try:
            insights = {
                'key_findings': [],
                'recommendations': [],
                'risk_factors': [],
                'opportunities': [],
                'data_reliability': ''
            }
            
            # 데이터 신뢰성 평가
            quality_score = quality_report.get('quality_score', 0)
            if quality_score >= 0.8:
                insights['data_reliability'] = '높음 - 분석 결과를 신뢰할 수 있습니다.'
            elif quality_score >= 0.6:
                insights['data_reliability'] = '보통 - 결과 해석시 주의가 필요합니다.'
            else:
                insights['data_reliability'] = '낮음 - 추가 데이터 수집이 필요합니다.'
            
            # 주요 발견사항
            if 'market_analysis' in preference_results:
                market_analysis = preference_results['market_analysis']
                dominant_brand = market_analysis.get('dominant_brand')
                if dominant_brand:
                    insights['key_findings'].append(f'{dominant_brand}이(가) 시장을 주도하고 있습니다.')
                
                # 시장 집중도
                hhi = market_analysis.get('market_concentration_hhi', 0)
                if hhi > 0.3:
                    insights['key_findings'].append('시장 집중도가 높습니다 (독과점 상태).')
                elif hhi < 0.15:
                    insights['key_findings'].append('시장이 균등하게 분산되어 있습니다.')
            
            # 계절성 관련 인사이트
            if 'brand_seasonality' in preference_results:
                brand_seasonality = preference_results['brand_seasonality']
                
                # 계절성이 강한 브랜드 식별
                for brand, analysis in brand_seasonality.get('brand_analysis', {}).items():
                    seasonality_metrics = analysis.get('seasonality_metrics', {})
                    cv = seasonality_metrics.get('seasonal_cv', 0)
                    
                    if cv > 0.5:
                        insights['key_findings'].append(f'{brand}은(는) 강한 계절성을 보입니다.')
                        insights['recommendations'].append(f'{brand}의 계절별 재고 및 마케팅 전략을 차별화하세요.')
            
            # 통계적 유의성 기반 인사이트
            if 'comprehensive_tests' in statistical_results:
                comprehensive_tests = statistical_results['comprehensive_tests']
                independence_tests = comprehensive_tests.get('independence_tests', {})
                
                if independence_tests.get('chi_square', {}).get('significant', False):
                    effect_size = independence_tests['chi_square'].get('effect_size', 0)
                    if effect_size > 0.3:
                        insights['key_findings'].append('브랜드와 계절성 간에 강한 연관성이 있습니다.')
                        insights['opportunities'].append('계절별 브랜드 특화 전략으로 시장 점유율을 확대할 수 있습니다.')
            
            # 위험 요소 식별
            if quality_score < 0.7:
                insights['risk_factors'].append('데이터 품질이 낮아 분석 결과의 신뢰성이 제한적입니다.')
            
            # 기본 권장사항
            if not insights['recommendations']:
                insights['recommendations'].append('데이터를 기반으로 한 의사결정을 계속 진행하세요.')
                insights['recommendations'].append('정기적인 데이터 품질 모니터링을 수행하세요.')
            
            return insights
            
        except Exception as e:
            logger.error(f"비즈니스 인사이트 생성 중 오류 발생: {str(e)}")
            return {
                'key_findings': ['인사이트 생성 중 오류가 발생했습니다.'],
                'recommendations': ['기본적인 데이터 검토를 수행하세요.'],
                'risk_factors': [],
                'opportunities': [],
                'data_reliability': '알 수 없음'
            }


# Flask 라우트 함수들
def register_preference_analysis_routes(app):
    """Flask 앱에 선호도 분석 라우트 등록"""
    
    analysis_api = PreferenceAnalysisAPI()
    
    @app.route('/api/analysis/preference-by-period', methods=['GET'])
    def preference_by_period():
        """월별/계절별 선호도 분석 엔드포인트"""
        try:
            year = request.args.get('year', None)
            period_type = request.args.get('period_type', 'month')
            
            result = analysis_api.analyze_preference_by_period(year, period_type)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"API 엔드포인트 오류: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'message': str(e)
            }), 500
    
    @app.route('/api/analysis/brand-seasonality', methods=['GET'])
    def brand_seasonality_detailed():
        """브랜드별 상세 계절성 분석 엔드포인트"""
        try:
            brands = request.args.getlist('brands')  # 복수 브랜드 지원
            
            result = analysis_api.analyze_brand_seasonality_detailed(brands if brands else None)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"API 엔드포인트 오류: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'message': str(e)
            }), 500
    
    @app.route('/api/analysis/model-seasonality', methods=['GET'])
    def model_seasonality_detailed():
        """모델별 상세 계절성 분석 엔드포인트"""
        try:
            top_n = int(request.args.get('top_n', 10))
            
            result = analysis_api.analyze_model_seasonality_detailed(top_n)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"API 엔드포인트 오류: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'message': str(e)
            }), 500