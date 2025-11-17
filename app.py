from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
from flask_restful import Api, Resource
import json
from sqlalchemy import text
from typing import Union, Optional, Any
from src.data_loader import get_data_from_db, get_db_connection
from src.simple_preference_analysis import create_simple_preference_api
from src.simple_trend_analysis import create_simple_trend_api
from src.services.daily_forecast import create_daily_forecast_api
from src.services.region_clustering import create_region_clustering_api

# --- Swagger ---
from flasgger import Swagger, swag_from

app = Flask(__name__)

# Swagger UI 설정
app.config["SWAGGER"] = {
    "title": "KUNI 2thecore Data Analysis API",
    "uiversion": 3
}
swagger = Swagger(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
CORS(app)
api = Api(app)


class DataAnalysisAPI(Resource):
    @swag_from({
        'responses': {
            200: {
                'description': 'API 기본 정보 확인',
                'examples': {
                    'application/json': {
                        "message": "KUNI 2thecore Data Analysis API",
                        "status": "running"
                    }
                }
            }
        }
    })
    def get(self):
        return {
            "message": "KUNI 2thecore Data Analysis API",
            "status": "running",
            "endpoints": {
                "/": "API 정보",
                "/api/data": "데이터 조회 (POST)",
                "/api/health": "헬스 체크",
                "/api/analysis/period": "선호도 분석 (GET)",
                "/api/analysis/trend": "연도별 트렌드 분석 (GET)"
            }
        }


class DataQueryAPI(Resource):
    @swag_from({
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'required': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'}
                    },
                    'example': {'query': 'SELECT * FROM table_name'}
                }
            }
        ],
        'responses': {
            200: {'description': '쿼리 실행 결과 반환'},
            400: {'description': '잘못된 요청'},
            500: {'description': '서버 오류'}
        }
    })
    def post(self):
        try:
            data = request.get_json()
            if not data or 'query' not in data:
                return {"error": "쿼리가 필요합니다. 'query' 필드를 포함해주세요."}, 400
            
            query = data['query']
            result_df = get_data_from_db(query)
            
            if result_df is not None:
                result_dict = result_df.to_dict('records')
                return {
                    "success": True,
                    "data": result_dict,
                    "row_count": len(result_dict)
                }
            else:
                return {"error": "쿼리 실행 중 오류가 발생했습니다."}, 500
                
        except Exception as e:
            return {"error": f"요청 처리 중 오류가 발생했습니다: {str(e)}"}, 500


class HealthCheckAPI(Resource):
    @swag_from({
        'responses': {
            200: {'description': 'DB 연결 상태 정상'},
            503: {'description': 'DB 연결 실패'}
        }
    })
    def get(self):
        try:
            engine = get_db_connection()
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "database": "connected",
                "message": "시스템이 정상적으로 작동 중입니다."
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }, 503


# Base API 클래스 - 공통 패턴 추상화
class BaseAnalysisAPI(Resource):
    """분석 API들의 공통 패턴을 처리하는 베이스 클래스"""

    def execute_analysis(self, analyzer_class, analyzer_module, method_name='analyze', **kwargs):
        """공통 분석 실행 패턴"""
        try:
            # 동적 임포트
            module = __import__(analyzer_module, fromlist=[analyzer_class])
            analyzer_cls = getattr(module, analyzer_class)

            # 분석기 인스턴스 생성 및 실행
            analyzer = analyzer_cls()
            method = getattr(analyzer, method_name)
            result = method(**kwargs)

            return result
        except Exception as e:
            app.logger.error(f'Analysis error: {str(e)}')
            return {"success": False, "error": f"분석 중 오류가 발생했습니다: {str(e)}"}, 500

    def get_param(self, param_name: str, default: Optional[Any] = None, param_type: type = str) -> Any:
        """파라미터 추출 및 타입 변환 헬퍼"""
        try:
            value = request.args.get(param_name, default)
            if value is None:
                return default
            if param_type == bool:
                return value.lower() == 'true'
            return param_type(value)
        except (ValueError, TypeError):
            return default


# 간소화된 선호도 분석 API 클래스
class PreferenceAnalysisAPI(BaseAnalysisAPI):
    @swag_from({'responses': {200: {'description': '선호도 분석 결과 반환'}}})
    def get(self):
        year = self.get_param('year')
        period_type = self.get_param('period_type', 'month')

        return self.execute_analysis(
            'SimplePreferenceAnalyzer',
            'src.simple_preference_analysis',
            'analyze_preferences',
            year=year,
            period_type=period_type
        )


# 간소화된 트렌드 분석 API 클래스
class TrendAnalysisAPI(BaseAnalysisAPI):
    @swag_from({'responses': {200: {'description': '연도별 트렌드 분석 결과 반환'}}})
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


class DailyForecastAPI(BaseAnalysisAPI):
    @swag_from({'responses': {200: {'description': '일일 예측 결과 반환'}}})
    def get(self):
        start_date = self.get_param('start_date')
        end_date = self.get_param('end_date')
        forecast_days = self.get_param('forecast_days', 7, int)

        return self.execute_analysis(
            'DailyForecastAnalyzer',
            'src.services.daily_forecast',
            'analyze',
            start_date=start_date,
            end_date=end_date,
            forecast_days=forecast_days
        )


class RegionClusteringAPI(BaseAnalysisAPI):
    @swag_from({'responses': {200: {'description': '지역 군집화 결과 반환'}}})
    def get(self):
        start_date = self.get_param('start_date')
        end_date = self.get_param('end_date')
        k = self.get_param('k', 5, int)
        use_end_points = self.get_param('use_end_points', 'true', bool)

        return self.execute_analysis(
            'RegionClusteringAnalyzer',
            'src.services.region_clustering',
            'analyze',
            start_date=start_date,
            end_date=end_date,
            k=k,
            use_end_points=use_end_points
        )


# 엔드포인트 등록
api.add_resource(DataAnalysisAPI, '/')
api.add_resource(DataQueryAPI, '/api/data')
api.add_resource(HealthCheckAPI, '/api/health')
api.add_resource(PreferenceAnalysisAPI, '/api/analysis/period')
api.add_resource(TrendAnalysisAPI, '/api/analysis/trend')
api.add_resource(DailyForecastAPI, '/api/forecast/daily')
api.add_resource(RegionClusteringAPI, '/api/clustering/regions')


# 에러 핸들러
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)