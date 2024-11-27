import requests
import pandas as pd

def fetch_carbon_data(api_url, api_key, output_file):
    """
    Fetch carbon emissions data from an API and save it to a CSV file.

    :param api_url: str, API endpoint URL
    :param api_key: str, API authentication key
    :param output_file: str, Path to save the output CSV
    """
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)  # Convert JSON to DataFrame
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")

# Example usage
if __name__ == "__main__":
    API_URL = "https://mock.api.com/carbon"  # Replace with the actual API URL
    API_KEY = "your_api_key"  # Replace with a valid API key
    OUTPUT_FILE = "raw_carbon_data.csv"
    fetch_carbon_data(API_URL, API_KEY, OUTPUT_FILE)


def transform_data(input_file, output_file):
    """
    Transform raw carbon data by cleaning and adding derived fields.

    :param input_file: str, Path to the raw data CSV
    :param output_file: str, Path to save the processed CSV
    """
    df = pd.read_csv(input_file)
    
    # Normalize data
    df.rename(columns={"emission_value": "emission", "activity_amount": "activity"}, inplace=True)
    
    # Add derived field: emission per activity unit
    df['emission_per_unit'] = df['emission'] / df['activity']
    
    # Handle missing data
    df.fillna(0, inplace=True)
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Save transformed data
    df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")

# Transform data
transform_data("raw_carbon_data.csv", "processed_carbon_data.csv")


from sqlalchemy import create_engine

def load_data_to_sql(file_path, db_url, table_name):
    """
    Load processed carbon data into a PostgreSQL database.

    :param file_path: str, Path to the processed data CSV
    :param db_url: str, Database connection URL
    :param table_name: str, Name of the table to insert data into
    """
    engine = create_engine(db_url)
    df = pd.read_csv(file_path)
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"Data loaded into table '{table_name}'")

# Load data into PostgreSQL
DB_URL = "postgresql://user:password@localhost:5432/carbon_footprint"
load_data_to_sql("processed_carbon_data.csv", DB_URL, "carbon_data")


from prophet import Prophet

def predict_emissions(data_file, output_file):
    """
    Predict future carbon emissions using a time-series model.

    :param data_file: str, Path to the processed data CSV
    :param output_file: str, Path to save the predictions
    """
    df = pd.read_csv(data_file)
    df = df[['date', 'emission']].rename(columns={'date': 'ds', 'emission': 'y'})
    
    # Train the model
    model = Prophet()
    model.fit(df)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    # Save predictions
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Predict emissions
predict_emissions("processed_carbon_data.csv", "emission_forecast.csv")


import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Load Data
historical_data = pd.read_csv("processed_carbon_data.csv")
forecast_data = pd.read_csv("emission_forecast.csv")

# Initialize Dash App
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Carbon Footprint Dashboard", style={"textAlign": "center"}),

    # Historical Emissions Graph
    html.Div([
        html.H2("Historical Emissions"),
        dcc.Graph(
            figure=px.line(
                historical_data, 
                x="date", 
                y="emission", 
                title="Historical Carbon Emissions",
                labels={"emission": "Emissions (kg)", "date": "Date"}
            )
        ),
    ]),

    # Forecasted Emissions Graph
    html.Div([
        html.H2("Forecasted Emissions"),
        dcc.Graph(
            figure=px.line(
                forecast_data, 
                x="ds", 
                y=["yhat", "yhat_lower", "yhat_upper"], 
                title="Forecasted Carbon Emissions",
                labels={"ds": "Date", "value": "Emissions (kg)"},
                line_group="variable"
            ).update_traces(mode='lines+markers')
        ),
    ]),
])

# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)


