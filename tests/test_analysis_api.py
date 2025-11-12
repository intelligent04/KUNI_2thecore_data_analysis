# tests/test_analysis_api.py
import pytest


def test_trend_analysis_default_params(client, mocker):
    """
    [RED] /api/analysis/trend 엔드포인트가 없는 상태에서
    이 테스트는 404 Not Found로 실패해야 합니다.
    """

    # --- 모킹 설정 (미래의 GREEN 상태를 위한 설정) ---
    # 우리가 테스트하려는 것은 'TrendAnalysisAPI'가 'SimpleTrendAnalyzer'를
    # '올바른 인자'로 호출하는가? 입니다.
    # 따라서 'SimpleTrendAnalyzer' 자체는 가짜(Mock)로 만듭니다.

    mock_analyzer_class = mocker.MagicMock()
    mock_analyzer_instance = mocker.MagicMock()

    # 'analyze_yearly_trend' 메서드가 반환할 가짜 결과
    mock_result = {"success": True, "data": "default_mock_data"}
    mock_analyzer_instance.analyze_yearly_trend.return_value = mock_result

    # 'SimpleTrendAnalyzer()' 생성자가 가짜 인스턴스를 반환하도록 설정
    mock_analyzer_class.return_value = mock_analyzer_instance

    # 'app.BaseAnalysisAPI.execute_analysis' 내부의 'getattr'을 모킹
    mocker.patch('app.getattr', return_value=mock_analyzer_class)
    mocker.patch('app.__import__')  # 실제 임포트 방지
    # --- 모킹 끝 ---

    # [When] 아직 존재하지 않는 API를 호출합니다.
    response = client.get('/api/analysis/trend')

    # [Then]
    # 이 테스트는 404 에러로 인해 아래 'assert 404 == 200'에서 실패할 것입니다.
    assert response.status_code == 200

    # 이 코드는 아직 실행되지 않습니다.
    json_data = response.get_json()
    assert json_data == mock_result

    # 'analyze_yearly_trend'가 기본값으로 호출되었는지 검증
    mock_analyzer_instance.analyze_yearly_trend.assert_called_once_with(
        start_year=2020,
        end_year=2025,
        top_n=5
    )