# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based data analysis project for KUNI 2thecore that provides data analysis reports for car rental companies. The project analyzes rental car usage patterns, brand preferences, seasonal trends, and location optimization using MySQL database connectivity and advanced data analysis capabilities.

### Business Purpose
The system provides data-driven insights to rental car companies using two main data sources:
- **car** table: Vehicle information including model, brand, year, type, status, and location
- **drivelog** table: Driving records with trip details, timestamps, locations, and events

### Core Services Provided
1. **Seasonal Brand/Vehicle Preference Analysis** - Monthly and seasonal analysis of preferred brands and vehicles with seasonality evaluation
2. **Trend Analysis** - Year-over-year changes in brand and vehicle preferences 
3. **Daily Vehicle Usage Forecasting** - Daily operational vehicle count analysis with SARIMA time series forecasting (1 week to 1 month ahead)
4. **Location Clustering Analysis** - Regional importance quantification for optimal rental location placement

## Environment Setup

The project requires a virtual environment with dependencies installed from `requirements.txt`:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

## Database Configuration

The project uses MySQL with SQLAlchemy and requires a `.env` file with database credentials:
- DB_HOST
- DB_USER  
- DB_PASSWORD
- DB_NAME
- DB_PORT

Database connection and data loading functionality is centralized in `src/data_loader.py`.

### Database Schema

#### car table
- `car_id` (Primary Key): Unique vehicle identifier
- `model`: Vehicle model name (e.g., K3, 투싼, 아이오닉 5)
- `brand`: Manufacturer (기아, 현대, 제네시스)
- `status`: Current status (IDLE, MAINTENANCE)
- `car_year`: Manufacturing year
- `car_type`: Vehicle category (소형, 중형, 대형, SUV)
- `car_number`: License plate number
- `sum_dist`: Total distance traveled (km)
- `login_id`: Associated user/company ID
- `last_latitude`, `last_longitude`: Current GPS coordinates

#### drivelog table  
- `drive_log_id` (Primary Key): Unique trip identifier
- `car_id` (Foreign Key): References car table
- `drive_dist`: Trip distance (km)
- `start_point`, `end_point`: Location names
- `start_latitude`, `start_longitude`: Trip origin coordinates
- `end_latitude`, `end_longitude`: Trip destination coordinates  
- `start_time`, `end_time`: Trip timestamps
- `created_at`: Record creation timestamp
- `model`, `brand`: Denormalized vehicle information
- `memo`: Trip notes/events (급감속, 과열 경고, etc.)
- `status`: Trip status

## Project Architecture

### Core Infrastructure
- `app.py` - Main Flask application with REST API endpoints and Swagger documentation
- `run_server.py` - Production server runner script 
- `src/data_loader.py` - Database connection and data loading utilities
  - `get_db_connection()` - Creates SQLAlchemy engine from .env credentials
  - `get_data_from_db(query)` - Executes SQL queries and returns pandas DataFrames
- `verify_setup.py` - Environment verification script for testing library imports

### Analysis Modules
- `src/simple_preference_analysis.py` - Seasonal brand/vehicle preference analysis with sklearn
- `src/simple_trend_analysis.py` - Year-over-year trend analysis 
- `src/services/daily_forecast.py` - Daily usage forecasting with SARIMA time series models
- `src/services/region_clustering.py` - Geographic clustering analysis
- `src/utils/cache.py` - Result caching utilities for performance optimization

### Supporting Files
- `src/data_quality.py` - Data validation and quality checks
- `src/visualization_enhanced.py` - Advanced visualization utilities
- `src/statistical_tests.py` - Statistical analysis tools
- `src/utils/font_config.py` - Cross-platform Korean font configuration for matplotlib
- `requirements.txt` - Full data science stack dependencies
- `ubuntu_setup.sh` - Ubuntu server deployment setup script

## Development Workflow

Common development commands for this data analysis project:

```bash
# Verify environment setup (import test)
python verify_setup.py

# Test database connection and data loading
python src/data_loader.py

# Start Flask web server (development)
python app.py
# or (preferred production runner)
python run_server.py

# Start Jupyter for interactive analysis
jupyter lab

# Test individual analysis modules
python -m src.simple_preference_analysis
python -m src.simple_trend_analysis
python -m src.services.daily_forecast
python -m src.services.region_clustering
```

## Flask Web Server

The project includes a Flask-based REST API server for data analysis:

- **Main Application**: `app.py` - Core Flask application with REST API endpoints
- **Server Runner**: `run_server.py` - Development server startup script
- **Base URL**: `http://localhost:5000`

### Available API Endpoints

**Core System Endpoints:**
- `GET /` - API information and available endpoints  
- `POST /api/data` - Execute SQL queries and return results as JSON
- `GET /api/health` - Health check and database connectivity status

**Data Analysis Endpoints:**
- `GET /api/analysis/period` - Monthly/seasonal brand and vehicle preference analysis (implemented in `src/simple_preference_analysis.py`)
- `GET /api/analysis/trend` - Year-over-year brand/vehicle preference trends (implemented in `src/simple_trend_analysis.py`)
- `GET /api/forecast/daily` - Daily vehicle usage counts with forecasting (implemented in `src/services/daily_forecast.py`)
- `GET /api/clustering/regions` - Regional importance analysis for location optimization (implemented in `src/services/region_clustering.py`)

**Note:** All analysis endpoints use Swagger documentation via flasgger. Access interactive API docs at `/apidocs/` when server is running.

### API Usage Example

```bash
# Health check
curl http://localhost:5000/api/health

# Execute a custom SQL query
curl -X POST http://localhost:5000/api/data \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM car LIMIT 5;"}'

# Get seasonal preference analysis
curl http://localhost:5000/api/analysis/period

# Get trend analysis 
curl http://localhost:5000/api/analysis/trend

# Get daily usage forecast
curl http://localhost:5000/api/forecast/daily

# Access Swagger documentation
# Open browser to: http://localhost:5000/apidocs/
```

## Data Analysis Services

### 1. Seasonal Brand/Vehicle Preference Analysis
**Purpose**: Analyze monthly and seasonal patterns in brand and vehicle preferences
- Calculate preference ratios by month/season for each brand (기아, 현대, 제네시스)
- Evaluate seasonality strength using statistical measures
- Identify peak seasons for specific vehicle types (소형, 중형, 대형, SUV)
- Generate seasonal preference rankings and trends

**Key Metrics**:
- Monthly/seasonal rental frequency by brand and model
- Seasonality index calculation
- Brand market share variations across seasons
- Vehicle type preference patterns

### 2. Year-over-Year Trend Analysis  
**Purpose**: Track long-term changes in brand and vehicle preferences over multiple years
- Compare brand preference shifts year-over-year
- Identify emerging trends in vehicle type preferences
- Analyze market share evolution for each manufacturer
- Detect preference pattern changes over time

**Key Metrics**:
- Annual growth rates by brand
- Market share trend analysis
- Vehicle type adoption patterns
- Preference volatility measurements

### 3. Daily Vehicle Usage Forecasting
**Purpose**: Provide operational insights and future demand prediction
- Analyze daily operational vehicle counts over specified periods
- Generate visual graphs of usage patterns
- Implement SARIMA time series forecasting for 1-week to 1-month ahead
- Create interactive dashboards with forecast visualizations

**Key Features**:
- Real-time daily usage tracking
- Seasonal adjustment in forecasting models
- Confidence intervals for predictions
- Visual trend analysis with matplotlib/seaborn

### 4. Location Clustering and Regional Analysis
**Purpose**: Optimize rental location placement through data-driven regional analysis
- Perform clustering analysis on trip start/end coordinates
- Quantify regional importance based on usage density
- Identify optimal locations for new rental stations
- Analyze geographical usage patterns and accessibility

**Key Techniques**:
- K-means clustering on geographical coordinates
- Regional demand density analysis
- Distance-based accessibility scoring
- Heatmap visualizations for location insights

## Key Libraries and Dependencies

The project includes a full data science stack:
- **Database**: SQLAlchemy, mysql-connector-python, python-dotenv
- **Data Analysis**: pandas, numpy, scipy, statsmodels
- **Machine Learning**: scikit-learn (clustering, regression, preprocessing)
- **Visualization**: matplotlib, seaborn (base64 encoding for API responses)
- **Web Server**: Flask, Flask-CORS, Flask-RESTful, flasgger (Swagger), Werkzeug
- **Jupyter**: jupyterlab, ipykernel for interactive development
- **Development**: All supporting libraries for data analysis workflows

## Key Implementation Notes

- **Cross-Platform Korean Font Support**: matplotlib configured with Korean fonts (Malgun Gothic for Windows, Noto Sans CJK KR for Linux, Apple SD Gothic Neo for macOS)
- **Result Caching**: Analysis endpoints use `@cache_result` decorator for performance
- **Base64 Visualizations**: Charts returned as base64 strings for API consumption
- **Error Handling**: Comprehensive error handling with structured JSON responses
- **Swagger Documentation**: All endpoints documented and accessible at `/apidocs/`
- **Database Requirements**: Requires `.env` file with MySQL connection parameters

## Ubuntu Server Deployment

For Ubuntu server deployment, use the provided setup script:

```bash
# Make setup script executable
chmod +x ubuntu_setup.sh

# Run Ubuntu setup script
./ubuntu_setup.sh
```

### Ubuntu Prerequisites

The setup script will install:
- **Korean Fonts**: fonts-noto-cjk, fonts-nanum, fonts-liberation
- **System Packages**: python3, python3-pip, python3-venv, build-essential
- **MySQL Client**: libmysqlclient-dev for database connectivity

### Manual Ubuntu Setup (Alternative)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y build-essential libmysqlclient-dev pkg-config

# Install Korean fonts (Critical for chart generation)
sudo apt install -y fonts-noto-cjk fonts-nanum fonts-liberation
sudo fc-cache -fv

# Verify Korean fonts are installed
fc-list :lang=ko

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file with database credentials
# DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Run server
python run_server.py
```

### Production Deployment

For production servers, use gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run production server
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or with supervisor for process management
sudo apt install supervisor
# Configure supervisor with provided config
```

**Important**: Korean font installation is critical for proper chart generation. Without Korean fonts, chart text will appear as squares or question marks.