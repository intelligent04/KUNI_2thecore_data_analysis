import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def get_db_connection():
    """
    .env 파일의 정보를 이용해 데이터베이스 커넥션 엔진을 생성
    """
    load_dotenv() # .env 파일에서 환경 변수 불러옴

    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    db_port = os.getenv("DB_PORT")

    # SQLAlchemy를 이용한 데이터베이스 연결 문자열 생성
    database_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    engine = create_engine(database_url)
    return engine

def get_data_from_db(query):
    """
    주어진 쿼리를 실행하여 데이터베이스에서 데이터를 가져와 Pandas DataFrame으로 반환
    """
    engine = get_db_connection()
    try:
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # 이 파일을 직접 실행할 경우 아래 코드가 동작 (테스트용)
    print("============================== 데이터베이스 연결 테스트 ==============================")

    # Check available tables
    tables_query = "SHOW TABLES;"
    tables_df = get_data_from_db(tables_query)
    if tables_df is not None:
        print("Available tables:")
        print(tables_df)
        print()

    test_query = "SELECT * FROM car LIMIT 10;"

    sample_df = get_data_from_db(test_query)

    if sample_df is not None:
        print("데이터 조회 성공!")
        print(sample_df)
