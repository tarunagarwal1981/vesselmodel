import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sqlalchemy import create_engine
from database import get_db_engine

# Database setup
def get_db_connection():
    return get_db_engine()

# Sidebar model selection
st.sidebar.header("Model Selection and Vessel Details")
model_options = ["Linear Regression with Polynomial Features", "Random Forest", "MLP Regressor"]
selected_model = st.sidebar.selectbox("Select a model to train:", model_options)

# User Inputs for Vessel Particulars
lpp = st.sidebar.number_input("Lpp (m)", min_value=50.0, max_value=400.0, step=0.1)
breadth = st.sidebar.number_input("Breadth (m)", min_value=10.0, max_value=100.0, step=0.1)
depth = st.sidebar.number_input("Depth (m)", min_value=5.0, max_value=50.0, step=0.1)
deadweight = st.sidebar.number_input("Deadweight (tons)", min_value=1000, max_value=500000, step=100)
year_of_built = st.sidebar.number_input("Year of Built", min_value=1900, max_value=2025, step=1)
vessel_type = st.sidebar.selectbox("Vessel Type", ["BULK CARRIER", "CONTAINER", "OIL TANKER"])
main_engine_make = st.sidebar.text_input("Main Engine Make")
main_engine_model = st.sidebar.text_input("Main Engine Model")
mcr = st.sidebar.number_input("MCR of Main Engine (kW)", min_value=500, max_value=100000, step=100)

# Function to get similar vessels from database
def get_similar_vessels(engine, lpp, breadth, depth, deadweight, vessel_type):
    query = """
    SELECT * FROM hull_particulars
    WHERE
        length_between_perpendiculars_m BETWEEN %(lpp_min)s AND %(lpp_max)s AND
        breadth_moduled_m BETWEEN %(breadth_min)s AND %(breadth_max)s AND
        depth BETWEEN %(depth_min)s AND %(depth_max)s AND
        deadweight BETWEEN %(deadweight_min)s AND %(deadweight_max)s AND
        vessel_type = %(vessel_type)s
    """
    params = {
        'lpp_min': lpp * 0.95,
        'lpp_max': lpp * 1.05,
        'breadth_min': breadth * 0.95,
        'breadth_max': breadth * 1.05,
        'depth_min': depth * 0.95,
        'depth_max': depth * 1.05,
        'deadweight_min': deadweight * 0.95,
        'deadweight_max': deadweight * 1.05,
        'vessel_type': vessel_type
    }
    try:
        print(f"Executing query: {query}")
        print(f"With parameters: {params}")
        return pd.read_sql(query, engine, params=params)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()
    params = {
        'lpp_min': lpp * 0.95,
        'lpp_max': lpp * 1.05,
        'breadth_min': breadth * 0.95,
        'breadth_max': breadth * 1.05,
        'depth_min': depth * 0.95,
        'depth_max': depth * 1.05,
        'deadweight_min': deadweight * 0.95,
        'deadweight_max': deadweight * 1.05,
        'vessel_type': vessel_type
    }
    return pd.read_sql(query, engine, params=params)
    return pd.read_sql(query, engine)

# Function to get speed, consumption, power data for selected vessels
def get_speed_consumption_data(engine, vessel_ids):
    query = f"""
    SELECT * FROM speed_consumption_data
    WHERE vessel_id IN ({', '.join(map(str, vessel_ids))})
    """
    return pd.read_sql(query, engine)

# Button to start fetching data and training
if st.sidebar.button("Fetch Data and Train Model"):
    engine = get_db_connection()
    similar_vessels = get_similar_vessels(engine, lpp, breadth, depth, deadweight, vessel_type)
    
    if similar_vessels.empty:
        st.write("No vessels found matching the given criteria.")
    else:
        st.write(f"Found {len(similar_vessels)} vessels matching the criteria.")
        st.write("Names of similar vessels:")
        st.write(similar_vessels['vessel_name'].tolist())
        vessel_ids = similar_vessels['imo'].tolist()
        speed_consumption_data = get_speed_consumption_data(engine, vessel_ids)
        
        if speed_consumption_data.empty:
            st.write("No speed, consumption, or power data available for the selected vessels.")
        else:
            st.write("Data fetched successfully. Ready for model training.")
            
            # Placeholder for model training code
            st.write("Model training functionality is currently hashed out.")

            # Placeholder for output table
            st.write("Creating output table (Speed, Power, Consumption, Displacement)...")
            # Add code to create and display output table here

# Placeholder for future steps and more details
st.sidebar.write("Once the model is trained, we will generate output tables and further analyze results.")
