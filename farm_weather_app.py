import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import io
import numpy as np
import os
import json
import concurrent.futures
import threading
from typing import Dict, Optional

# Configure page
st.set_page_config(
    page_title="Farm Weather Dashboard",
    page_icon="🌧️",
    layout="wide"
)

# Farm stations data
STATIONS = [
    {"Farm": "Delta", "Location": "Ayr Research Station", "StationID": 33002},
    {"Farm": "Barratta", "Location": "Kalamai", "StationID": 33035},
    {"Farm": "Haughton", "Location": "Giru Post Office", "StationID": 33028},
    {"Farm": "Boolarwell", "Location": "Talwood State School", "StationID": 42027},
    {"Farm": "South Callandoon", "Location": "Goondiwindi Airport", "StationID": 41521},
    {"Farm": "Gunedra", "Location": "Wee Waa (George St)", "StationID": 53044},
    {"Farm": "Enfield North", "Location": "Tia (Highrent)", "StationID": 57045},
    {"Farm": "Canomodine", "Location": "Canowindra", "StationID": 65006},
    {"Farm": "Springfield", "Location": "Boorowa", "StationID": 70220},
    {"Farm": "Meadowbank", "Location": "Koorawatha", "StationID": 73021},
    {"Farm": "Burrangong", "Location": "Young Airport", "StationID": 73138},
    {"Farm": "Watson Park", "Location": "Yarra (Rowe S Lagoon)", "StationID": 70088},
    {"Farm": "Minjah", "Location": "Hawkesdale", "StationID": 90045},
    {"Farm": "Cheviot Hills", "Location": "Penshurst", "StationID": 90063},
    {"Farm": "Rippling Water", "Location": "Ournie (Waralla)", "StationID": 72038},
    {"Farm": "Deltroit", "Location": "Mundarlo", "StationID": 73055},
    {"Farm": "Long Mountain", "Location": "Mudgee", "StationID": 62021},
]

# Create data directory if it doesn't exist
DATA_DIR = "farm_weather_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_station_data(station_id, end_date):
    """Fetch data from SILO API for a given station."""
    url = (
        f"https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php?"
        f"station={station_id}&start=19000101&finish={end_date.strftime('%Y%m%d')}"
        f"&format=csv&comment=rxn&username=tlmort91@gmail.com"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Read the response text and split into lines
        lines = response.text.strip().split('\n')
        
        # Skip the first few lines that contain metadata
        data_lines = []
        for line in lines:
            # Check if line contains data (has station ID as first value)
            parts = line.split(',')
            if len(parts) >= 9 and parts[0].isdigit():
                data_lines.append(line)
        
        if not data_lines:
            return None
        
        # Create a new StringIO object with just the data lines
        csv_data = '\n'.join(data_lines)
        
        # Parse CSV data
        df = pd.read_csv(
            io.StringIO(csv_data),
            names=['station', 'date', 'daily_rain', 'daily_rain_source', 'max_temp', 
                   'max_temp_source', 'min_temp', 'min_temp_source', 'metadata'],
            skiprows=0
        )
        
        # Clean and convert date column - handle different formats
        def parse_date(date_str):
            date_str = str(date_str).strip()
            if date_str in ['nan', 'NaN', '']:
                return pd.NaT
            
            # Try different date formats
            formats = ['%Y-%m-%d', '%Y%m%d', '%d/%m/%Y', '%d-%m-%Y']
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            
            # If all formats fail, try pandas automatic parsing
            try:
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        df['date'] = df['date'].apply(parse_date)
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Convert rain column, handling spaces and invalid values
        df['daily_rain'] = df['daily_rain'].astype(str).str.strip()
        df['daily_rain'] = pd.to_numeric(df['daily_rain'], errors='coerce')
        
        # Remove rows with invalid rainfall data
        df = df.dropna(subset=['daily_rain'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        print(f"Error fetching data for station {station_id}: {str(e)}")
        return None

def fetch_single_station(station: Dict, end_date: datetime.date, progress_callback=None) -> tuple:
    """Fetch data for a single station and return results."""
    station_id = station["StationID"]
    farm_name = station["Farm"]
    
    if progress_callback:
        progress_callback(f"Fetching {farm_name}...")
    
    try:
        df = fetch_station_data(station_id, end_date)
        if df is not None and len(df) > 0:
            # Save to file
            data_file = os.path.join(DATA_DIR, f"{station_id}_data.csv")
            metadata_file = os.path.join(DATA_DIR, f"{station_id}_metadata.json")
            
            df.to_csv(data_file, index=False)
            with open(metadata_file, 'w') as f:
                json.dump({'last_update': end_date.strftime('%Y-%m-%d')}, f)
            
            return farm_name, df, None
        else:
            return farm_name, None, f"No data returned for {farm_name}"
    except Exception as e:
        return farm_name, None, f"Error fetching {farm_name}: {str(e)}"

def get_last_refresh_dates() -> Dict[str, Optional[str]]:
    """Get the last refresh date for each station."""
    refresh_dates = {}
    
    for station in STATIONS:
        station_id = station["StationID"]
        farm_name = station["Farm"]
        metadata_file = os.path.join(DATA_DIR, f"{station_id}_metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    refresh_dates[farm_name] = metadata['last_update']
            except:
                refresh_dates[farm_name] = None
        else:
            refresh_dates[farm_name] = None
    
    return refresh_dates

def load_existing_data() -> Dict[str, pd.DataFrame]:
    """Load existing data without fetching new data."""
    all_data = {}
    
    for station in STATIONS:
        station_id = station["StationID"]
        farm_name = station["Farm"]
        data_file = os.path.join(DATA_DIR, f"{station_id}_data.csv")
        
        if os.path.exists(data_file):
            try:
                df = pd.read_csv(data_file)
                df['date'] = pd.to_datetime(df['date'])
                all_data[farm_name] = df
            except Exception as e:
                st.warning(f"Error loading existing data for {farm_name}: {str(e)}")
    
    return all_data

def refresh_data_concurrent(end_date: datetime.date) -> Dict[str, pd.DataFrame]:
    """Refresh data for all stations concurrently."""
    all_data = {}
    failed_stations = []
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Thread-safe counter for progress
    completed = {'count': 0}
    lock = threading.Lock()
    
    def update_progress(message):
        with lock:
            completed['count'] += 1
            progress = completed['count'] / len(STATIONS)
            progress_bar.progress(progress)
            status_text.text(f"{message} ({completed['count']}/{len(STATIONS)})")
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_station = {
            executor.submit(fetch_single_station, station, end_date, update_progress): station 
            for station in STATIONS
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_station):
            farm_name, df, error = future.result()
            
            if df is not None:
                all_data[farm_name] = df
            else:
                failed_stations.append(farm_name)
                if error:
                    st.warning(error)
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_stations:
        st.warning(f"Failed to update data for: {', '.join(failed_stations)}")
    
    st.success(f"Successfully refreshed data for {len(all_data)} farms")
    
    return all_data

def calculate_trailing_rainfall(df, date, periods):
    """Calculate trailing rainfall for specified periods."""
    results = {}
    
    # Ensure date is a pandas Timestamp for consistency
    if isinstance(date, datetime):
        end_date = pd.Timestamp(date)
    elif isinstance(date, (str, np.datetime64)):
        end_date = pd.Timestamp(date)
    else:
        end_date = pd.Timestamp(date)
    
    for period_months in periods:
        # Calculate start date as Timestamp
        start_date = end_date - pd.Timedelta(days=period_months * 30.44)
        
        # Ensure df['date'] is also pandas Timestamp
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter data for the period
        period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        total_rainfall = period_data['daily_rain'].sum()
        results[f"{period_months}_month"] = total_rainfall
    
    return results

def get_historical_data_for_period(df, reference_date, months):
    """Get historical data for the same period across all years."""
    historical_data = []
    
    # Ensure reference_date is a pandas Timestamp
    if isinstance(reference_date, datetime):
        reference_date = pd.Timestamp(reference_date)
    elif isinstance(reference_date, (str, np.datetime64)):
        reference_date = pd.Timestamp(reference_date)
    else:
        reference_date = pd.Timestamp(reference_date)
    
    current_year = reference_date.year
    
    # Calculate start and end dates for the period
    end_date = reference_date
    start_date = end_date - pd.Timedelta(days=months * 30.44)
    
    # Extract month and day from start and end dates
    start_month, start_day = start_date.month, start_date.day
    end_month, end_day = end_date.month, end_date.day
    
    # Ensure df['date'] is pandas Timestamp
    df['date'] = pd.to_datetime(df['date'])
    
    # Get data for each year
    min_year = df['date'].min().year
    for year in range(min_year, current_year + 1):
        # Handle year boundaries
        if start_date.year != end_date.year:
            # Period spans two years
            try:
                year_start = pd.Timestamp(year - 1, start_month, start_day)
                year_end = pd.Timestamp(year, end_month, end_day)
            except ValueError:
                # Handle leap year issues (e.g., Feb 29)
                year_start = pd.Timestamp(year - 1, start_month, min(start_day, 28))
                year_end = pd.Timestamp(year, end_month, min(end_day, 28))
        else:
            # Period within same year
            try:
                year_start = pd.Timestamp(year, start_month, start_day)
                year_end = pd.Timestamp(year, end_month, end_day)
            except ValueError:
                # Handle leap year issues (e.g., Feb 29)
                year_start = pd.Timestamp(year, start_month, min(start_day, 28))
                year_end = pd.Timestamp(year, end_month, min(end_day, 28))
        
        # Filter data for this year's period
        year_data = df[(df['date'] >= year_start) & (df['date'] <= year_end)]
        if len(year_data) > 0:
            total_rainfall = year_data['daily_rain'].sum()
            historical_data.append({
                'year': year,
                'rainfall': total_rainfall
            })
    
    return historical_data

def calculate_percentile(current_value, historical_values):
    """Calculate percentile of current value compared to historical values."""
    if len(historical_values) == 0:
        return None
    
    # Add current value to historical data for percentile calculation
    all_values = list(historical_values) + [current_value]
    percentile = (sum(1 for v in all_values if v <= current_value) / len(all_values)) * 100
    return round(percentile)

def create_rainfall_summary_table(all_data):
    """Create the main rainfall summary table."""
    yesterday = datetime.now().date() - timedelta(days=1)
    # Convert to pandas Timestamp for consistency
    yesterday = pd.Timestamp(yesterday)
    periods = [12, 6, 3, 2, 1]  # months
    
    summary_data = []
    
    for station in STATIONS:
        farm_name = station["Farm"]
        df = all_data.get(farm_name)
        
        if df is None:
            continue
        
        # Ensure the dataframe has the right date format
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        row_data = {"Farm": farm_name}
        
        # Calculate trailing rainfall totals
        trailing_data = calculate_trailing_rainfall(df, yesterday, periods)
        
        # Calculate percentiles
        for period in periods:
            rainfall = trailing_data[f"{period}_month"]
            
            # Get historical data for same period
            historical_data = get_historical_data_for_period(df, yesterday, period)
            historical_values = [d['rainfall'] for d in historical_data if d['year'] < yesterday.year]
            
            # Calculate percentile
            percentile = calculate_percentile(rainfall, historical_values)
            
            row_data[f"{period}_month_rain"] = rainfall
            row_data[f"{period}_month_percentile"] = percentile
        
        summary_data.append(row_data)
    
    return pd.DataFrame(summary_data)



def main():
    st.title("🌧️ MHPF Farm Weather Dashboard")
    st.markdown("Trailing rainfall totals and percentiles for our farms")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'all_data' not in st.session_state:
        st.session_state.all_data = {}
    
    # Get last refresh dates
    refresh_dates = get_last_refresh_dates()
    
    # Display refresh status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Find the most recent and oldest refresh dates
        valid_dates = [date for date in refresh_dates.values() if date is not None]
        if valid_dates:
            most_recent = max(valid_dates)
            oldest = min(valid_dates)
            farms_with_data = len(valid_dates)
            total_farms = len(STATIONS)
            
            if most_recent == oldest:
                st.info(f"📅 Data last refreshed: **{most_recent}** ({farms_with_data}/{total_farms} farms)")
            else:
                st.info(f"📅 Data last refreshed: **{oldest}** to **{most_recent}** ({farms_with_data}/{total_farms} farms)")
        else:
            st.warning("📅 No cached data found. Click 'Refresh Data' to fetch weather data.")
    
    with col2:
        yesterday = datetime.now().date() - timedelta(days=1)
        if st.button("🔄 Refresh Data", help=f"Update data to {yesterday.strftime('%Y-%m-%d')}"):
            with st.spinner("Refreshing weather data..."):
                st.session_state.all_data = refresh_data_concurrent(yesterday)
                st.session_state.data_loaded = True
                st.rerun()
    
    # Load existing data on first load
    if not st.session_state.data_loaded:
        st.session_state.all_data = load_existing_data()
        st.session_state.data_loaded = True
    
    # Debug section (expandable)
    with st.expander("Debug Information", expanded=False):
        st.write("**Refresh Status by Farm:**")
        for farm, date in refresh_dates.items():
            status = f"✅ {date}" if date else "❌ No data"
            st.write(f"- {farm}: {status}")
        
        if st.button("Test API Connection"):
            st.write("Testing connection to SILO API...")
            test_url = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php?station=33002&start=20240101&finish=20240105&format=csv&comment=rxn&username=tlmort91@gmail.com"
            try:
                response = requests.get(test_url, timeout=10)
                st.write(f"Status Code: {response.status_code}")
                st.write("First 500 characters of response:")
                st.text(response.text[:500])
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Create main rainfall table
    st.header("Rainfall Summary")
    
    if st.session_state.all_data:
        df = create_rainfall_summary_table(st.session_state.all_data)
        
        if not df.empty:
            # Show data date range
            yesterday = datetime.now().date() - timedelta(days=1)
            st.markdown(f"*Analysis period ending: {yesterday.strftime('%Y-%m-%d')}*")
            
            # Add export button
            col1, col2 = st.columns([4, 1])
            with col2:
                # Create Excel file in memory
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Rainfall Summary', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Export to Excel",
                    data=excel_data,
                    file_name=f"rainfall_summary_{yesterday.strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Display as Streamlit dataframe
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Additional features
            st.header("Additional Analysis")
            
            # Farm selector for detailed analysis
            available_farms = [farm for farm in [station["Farm"] for station in STATIONS] if farm in st.session_state.all_data]
            
            if available_farms:
                selected_farm = st.selectbox("Select farm for detailed analysis:", available_farms)
                
                if selected_farm:
                    st.subheader(f"Detailed Analysis - {selected_farm}")
                    
                    # Period selector
                    period_months = st.selectbox("Select period for detailed view:", 
                                               [1, 2, 3, 6, 12], index=2)
                    
                    # Get historical data for selected farm and period
                    farm_data = st.session_state.all_data[selected_farm]
                    yesterday = datetime.now().date() - timedelta(days=1)
                    # Convert to pandas Timestamp
                    yesterday = pd.Timestamp(yesterday)
                    historical_data = get_historical_data_for_period(farm_data, yesterday, period_months)
                    
                    if historical_data:
                        hist_df = pd.DataFrame(historical_data)
                        hist_df = hist_df.sort_values('year')
                        
                        # Add percentile for each year
                        hist_df['percentile'] = hist_df['rainfall'].rank(pct=True) * 100
                        hist_df['percentile'] = hist_df['percentile'].round()
                        
                        # Display table
                        st.dataframe(hist_df, hide_index=True)
                        
                        # Create bar chart
                        st.bar_chart(
                            hist_df.set_index('year')['rainfall'],
                            height=400,
                            use_container_width=True
                        )
            else:
                st.warning("No farm data available for detailed analysis.")
        else:
            st.error("No data available for display.")
    else:
        st.info("💡 Click the 'Refresh Data' button above to fetch weather data for all farms.")

if __name__ == "__main__":
    main()
