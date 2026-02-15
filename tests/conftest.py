# tests/conftest.py
# tests/conftest.py
import sys
import os

# 'tests' 폴더의 부모 폴더(프로젝트 루트)를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytest
from app import app as flask_app  # app.py에서 app 객체를 임포트

@pytest.fixture
def app():
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()