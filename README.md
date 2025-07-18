# AirSense Copenhagen

## Overview

**AirSense Copenhagen** is a real-time air quality monitoring project focused on identifying pollution trends across Copenhagen’s urban landscape. It uses API-driven data collection and scientifically standardized AQI calculations to support smarter, cleaner city planning.

This university project from our Machine Learning course helps advance Copenhagen’s carbon-neutral goals by turning air quality data into actionable insights, especially in relation to traffic patterns. The AQI calculations are based on official EPA methodology, ensuring consistent and comparable pollution scoring across locations.

## Key Features

- Integrates with the Open-Meteo Air Quality API  
- Monitors pollution data from multiple urban points in Copenhagen  
- Implements EPA-standard AQI computation logic  
- Applies correct pollutant-specific averaging rules:
  - 24-hour averages for PM2.5 and PM10
  - 8-hour averages for O3 and CO
  - 1-hour values for NO2 and SO2  
- Highlights the dominant pollutant and air quality rating per location  
- Categorizes pollution levels based on EPA AQI bands  

## Project Structure

```
airsense_copenhagen/
├── config.py             # Centralized configuration and constants
├── aqi_breakpoints.py    # AQI threshold data
├── aqi_calculator.py     # Core AQI logic
├── data_fetcher.py       # API querying and data collection
├── data_processor.py     # Data cleaning and aggregation
├── main.py               # Runs the full pipeline
└── README.md             # Documentation
```

### File Roles

- `config.py`: Coordinates, API keys, and unit conversion constants  
- `aqi_breakpoints.py`: EPA AQI breakpoint definitions  
- `aqi_calculator.py`: Calculates AQI from pollutant concentrations  
- `data_fetcher.py`: Downloads air quality data using Open-Meteo  
- `data_processor.py`: Applies averages and prepares AQI metrics  
- `main.py`: Ties everything together and produces final output  

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/airsense_copenhagen.git
   cd airsense_copenhagen
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install requests pandas numpy
   ```

## How to Run

Use the command below to fetch, process, and evaluate air quality data:

```bash
python main.py
```

This script will:
- Retrieve historical air quality data from three Copenhagen locations  
- Apply EPA-standard time-weighted averages  
- Compute AQI values per location and time interval  
- Output the most recent readings per area  

## Working with Results

Each location produces a DataFrame with these key columns:

- `time`: Measurement timestamp  
- `AQI`: Calculated index score  
- `AQI_Category`: Qualitative pollution level  
- `Dominant_Pollutant`: Pollutant contributing most to the AQI  

Example usage in Python:

```python
import pandas as pd
from main import main

# Run pipeline
dfs = main()

# Access one location
boulevard_df = dfs["H.C. Andersens Boulevard"]

# Filter last 30 days
recent_data = boulevard_df[boulevard_df['time'] > pd.Timestamp.now() - pd.Timedelta(days=30)]

# Average AQI
print("Avg AQI (last 30 days):", recent_data['AQI'].mean())
```

## Data Sources

We utilize the Open-Meteo Air Quality API for high-resolution hourly data on:

- PM2.5 / PM10  
- NO2  
- CO  
- SO2  
- O3  

### Locations Tracked

- **H.C. Andersens Boulevard** – Dense traffic/commercial area  
- **Nørrebro** – Residential and mixed-use district  
- **Amager Strandpark** – Coastal green space with low emissions  

## AQI Methodology

Following the EPA standard, we:

1. Compute pollutant-specific rolling averages  
2. Convert concentrations into AQI using breakpoint tables  
3. Use the highest AQI value per location as the overall AQI  
4. Determine the dominant pollutant  
5. Categorize results into Good, Moderate, Unhealthy, etc.

## Roadmap

Future enhancements may include:

- Integration with traffic congestion data  
- Machine learning to predict AQI trends  
- Web-based dashboard for live updates  
- Support for additional monitoring sites  
- Seasonal and temporal trend visualizations  

## Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for providing API access  
- U.S. Environmental Protection Agency for AQI framework  
