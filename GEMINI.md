# KUNI 2thecore Data Analysis API

## Project Overview

This project is a Python-based web application that provides a data analysis API. It uses Flask to create the web server and RESTful API endpoints. The analysis is performed using popular data science libraries like pandas, scikit-learn, and statsmodels. The API provides insights into car usage data, including preference analysis, trend analysis, daily forecasting, and region clustering. The analysis results, including charts and graphs generated with Matplotlib and Seaborn, are returned as JSON.

### Key Technologies

*   **Backend:** Python, Flask, Flask-RESTful
*   **Data Handling:** pandas, NumPy, SQLAlchemy
*   **Machine Learning/Statistics:** scikit-learn, statsmodels
*   **API Documentation:** Flasgger (Swagger UI)
*   **Database:** MySQL (inferred from `data_loader.py`)

### Architecture

The application is structured as follows:

*   `app.py`: The main entry point of the application. It defines the Flask app, API resources, and routes.
*   `run_server.py`: A script to run the Flask development server.
*   `src/`: This directory contains the core logic of the application.
    *   `data_loader.py`: Handles the database connection and data retrieval.
    *   `simple_preference_analysis.py`, `simple_trend_analysis.py`, etc.: Modules that contain the data analysis logic.
    *   `services/`: Contains more complex analysis services.
    *   `utils/`: Utility functions, such as font configuration for charts.
*   `requirements.txt`: Lists all the Python dependencies.
*   `.env`: (Not present, but inferred from `data_loader.py`) Contains the database connection credentials.

## Building and Running

### 1. Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```
2.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create a `.env` file** in the root directory with the following content, replacing the placeholder values with your actual database credentials:
    ```
    DB_HOST=your_db_host
    DB_USER=your_db_user
    DB_PASSWORD=your_db_password
    DB_NAME=your_db_name
    DB_PORT=your_db_port
    ```

### 2. Running the Application

To start the development server, run the following command:

```bash
python run_server.py
```

The application will be available at `http://localhost:5000`.

### 3. API Documentation

API documentation is available via Swagger UI at `http://localhost:5000/apidocs/`.

## Development Conventions

*   **Code Style:** The code follows the general Python conventions (PEP 8).
*   **Modularity:** The code is organized into modules based on functionality (e.g., data loading, different types of analysis).
*   **Error Handling:** The application includes error handlers for 500 and 404 errors. The analysis functions also have try-except blocks to handle potential errors during analysis.
*   **Logging:** The application uses the `logging` module to log information, warnings, and errors to a file (`app.log`) and the console.
*   **API Design:** The API is designed in a RESTful manner. It uses `GET` for retrieving data and `POST` for actions that involve sending data to the server (like executing a custom query).
