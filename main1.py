'''
import pandas as pd
import json

def extract_data_from_file(file_path):
    """
    Extracts electricity emissions data from a file for testing purposes.
    
    Parameters:
        file_path (str): Path to the test input file (JSON or CSV).
        
    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        if file_path.endswith(".json"):
            with open(file_path, "r") as file:
                data = json.load(file)
            df = pd.DataFrame(data)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type. Use a JSON or CSV file.")
        
        print("Data successfully extracted from file.")
        return df
    except Exception as e:
        print(f"Error during data extraction: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
'''

#Data Extraction (E)
#We’ll create a Python script to extract electricity consumption data from the API and save it to a CSV file for later processing.
import requests
import pandas as pd

# Function to fetch electricity carbon emission data from the API
def fetch_electricity_data(api_url, api_key, electricity_value, country, state=None, unit="mwh"):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Request body with electricity consumption details
    data = {
        "type": "electricity",
        "electricity_unit": unit,
        "electricity_value": electricity_value,
        "country": country,
        "state": state
    }
    
    # Make the POST request
    response = requests.post(api_url, json=data, headers=headers)
    
    # Check if the request was successful
    if response.status_code in (200,201):
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code} - - {response.text}")

# Example usage
if __name__ == "__main__":
    API_URL = "https://www.carboninterface.com/api/v1/estimates"
    API_KEY = ""  # Replace with your actual API key
    
    try:
        # Fetch the data
        response_data = fetch_electricity_data(API_URL, API_KEY, electricity_value=42, country="us", state="fl")
        
        # Extract the attributes part of the response
        attributes = response_data['data']['attributes']
        
        # Convert the attributes to a DataFrame
        df = pd.DataFrame([attributes])
        
        # Save the DataFrame to a CSV file
        df.to_csv("electricity_carbon_data.csv", index=False)
        print("Data successfully fetched and saved to electricity_carbon_data.csv")
    except Exception as e:
        print(str(e))


#Data Transformation (T)
#Now, let’s process the extracted data. We’ll clean it up and prepare it for analysis.
#Specifically, we’ll focus on extracting and formatting the relevant fields.

import pandas as pd

def transform_data(input_file, output_file):
    # Load the raw data
    data = pd.read_csv(input_file)
    
    # Clean up and format the data (e.g., convert to appropriate types)
    data['electricity_value'] = pd.to_numeric(data['electricity_value'], errors='coerce')
    data['carbon_g'] = pd.to_numeric(data['carbon_g'], errors='coerce')
    data['carbon_lb'] = pd.to_numeric(data['carbon_lb'], errors='coerce')
    data['carbon_kg'] = pd.to_numeric(data['carbon_kg'], errors='coerce')
    data['carbon_mt'] = pd.to_numeric(data['carbon_mt'], errors='coerce')
    data['estimated_at'] = pd.to_datetime(data['estimated_at'])
    
    # Save the transformed data to a new CSV file
    data.to_csv(output_file, index=False)
    print("Data transformation complete. Saved to transformed_carbon_data.csv")

# Example usage
transform_data("electricity_carbon_data.csv", "transformed_carbon_data.csv")

#Data Loading (L)
#We will now load the processed data into a SQlLite database. You’ll need a SQlLite database set up locally or remotely for this.
import sqlite3

def load_to_sqlite(transformed_df, database_name="carbon_emissions.db", table_name="electricity_emissions"):
    """
    Loads a transformed DataFrame into an SQLite database.
    
    Parameters:
        transformed_df (pd.DataFrame): The transformed DataFrame ready for loading.
        database_name (str): The name of the SQLite database file.
        table_name (str): The name of the table to store data.
    """
    try:
        # Connect to the SQLite database (creates the file if it doesn't exist)
        conn = sqlite3.connect(database_name)
        print(f"Connected to SQLite database: {database_name}")
        
        # Load the DataFrame into the database
        transformed_df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Data successfully loaded into table '{table_name}' in SQLite database.")
    
    except Exception as e:
        print(f"Error while loading data to SQLite: {e}")
    
    finally:
        # Ensure the connection is closed
        conn.close()
        print("SQLite connection closed.")
if __name__ == "__main__":
    data=pd.read_csv('transformed_carbon_data.csv')
    transformed_df = pd.DataFrame(data)
    # Call the load function
    load_to_sqlite(transformed_df)



#AI Model (Prophet)
#Next, let’s use the Prophet library to forecast future carbon emissions based on historical data.
from prophet import Prophet
import pandas as pd

# Function to predict carbon emissions
def predict_emissions(data_file):
    data = pd.read_csv(data_file)
    
    # Select the required columns for Prophet
    df = data[['estimated_at', 'carbon_g']]
    df.columns = ['ds', 'y']  # Prophet requires the columns to be named 'ds' (date) and 'y' (value)
    
    # Initialize and train the Prophet model
    model = Prophet()
    model.fit(df)
    
    # Create a future dataframe (e.g., predicting for the next 365 days)
    future = model.make_future_dataframe(df, periods=365)
    
    # Predict emissions
    forecast = model.predict(future)
    
    # Return the forecasted data
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Example usage
forecast = predict_emissions("transformed_carbon_data.csv")
forecast.to_csv("carbon_emission_forecast.csv", index=False)
print("Emissions forecast saved to carbon_emission_forecast.csv")


#Data Visualization (Dash)
#Now let’s build a Dash web app to visualize the historical and predicted carbon emissions.

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load historical and forecast data
historical_data = pd.read_csv("transformed_carbon_data.csv")
forecast_data = pd.read_csv("carbon_emission_forecast.csv")

# Create the Dash web app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Carbon Footprint Dashboard"),
    
    # Historical Emissions Graph
    dcc.Graph(
        figure=px.line(historical_data, x="estimated_at", y="carbon_g", title="Historical Carbon Emissions (g)")
    ),
    
    # Forecast Emissions Graph
    dcc.Graph(
        figure=px.line(forecast_data, x="ds", y="yhat", title="Forecasted Carbon Emissions (g)")
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)


