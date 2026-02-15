import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.simple_preference_analysis import SimplePreferenceAnalyzer

class TestSimplePreferenceAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = SimplePreferenceAnalyzer()
        # Create a sample DataFrame for testing
        self.sample_data = {
            'start_time': pd.to_datetime(['2023-01-15', '2023-04-20', '2023-07-10', '2023-10-05', '2023-01-25']),
            'brand': ['현대', '기아', '제네시스', '현대', '기아'],
            'model': ['Sonata', 'K5', 'G80', 'Avante', 'K3'],
            'car_type': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'Sedan']
        }
        self.sample_df = pd.DataFrame(self.sample_data)

    @patch('src.simple_preference_analysis.get_data_from_db')
    def test_load_data(self, mock_get_data_from_db):
        # Mock the database call
        mock_get_data_from_db.return_value = self.sample_df

        # Test with a specific year
        df = self.analyzer._load_data(year='2023')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 5)
        self.assertIn('year', df.columns)
        self.assertIn('month', df.columns)
        self.assertIn('season', df.columns)

        # Test without a specific year
        df_no_year = self.analyzer._load_data(year=None)
        self.assertIsInstance(df_no_year, pd.DataFrame)
        self.assertFalse(df_no_year.empty)

    def test_create_all_charts(self):
        # Use the sample DataFrame to test chart creation
        charts = self.analyzer._create_all_charts(self.sample_df, period_type='month')
        self.assertIsInstance(charts, dict)
        for chart_name, chart_data in charts.items():
            self.assertIsInstance(chart_data, str)
            self.assertTrue(chart_data.startswith('data:image/jpeg;base64,'))

    @patch('src.simple_preference_analysis.SimplePreferenceAnalyzer._load_data')
    def test_analyze_preferences(self, mock_load_data):
        # Mock the data loading
        mock_load_data.return_value = self.sample_df

        # Test the main analysis function
        result = self.analyzer.analyze_preferences(year='2023', period_type='month')
        self.assertTrue(result['success'])
        self.assertIn('visualizations', result)
        self.assertIsInstance(result['visualizations'], dict)
        self.assertIn('brand_period_heatmap', result['visualizations'])

    def test_color_for_brand(self):
        # Test predefined colors
        self.assertEqual(self.analyzer._color_for_brand('현대'), '#1f77b4')
        self.assertEqual(self.analyzer._color_for_brand('기아'), '#ff7f0e')

        # Test dynamic color assignment
        dynamic_color_1 = self.analyzer._color_for_brand('르노')
        dynamic_color_2 = self.analyzer._color_for_brand('쌍용')
        self.assertNotEqual(dynamic_color_1, dynamic_color_2)
        # Test that the same dynamic color is returned for the same brand
        self.assertEqual(self.analyzer._color_for_brand('르노'), dynamic_color_1)

if __name__ == '__main__':
    unittest.main()
