import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sqlalchemy import create_engine
from database import get_db_engine
import traceback

# ... (previous code remains the same)

# Function to get speed, consumption, power data for selected vessels
def get_vessel_performance_data(engine, vessel_names):
    query_ballast = """
    SELECT VESSEL_NAME, speed_knts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s AND load_type = 'Ballast'
    """
    query_scantling = """
    SELECT VESSEL_NAME, speed_knts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s AND load_type = 'Scantling'
    """
    query_design = """
    SELECT VESSEL_NAME, speed_knts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s AND load_type = 'Design'
    """
    params = {
        'vessel_names_list': tuple([name.upper() for name in vessel_names])
    }
    
    try:
        st.write(f"Executing query for Ballast data with parameters: {params}")
        df_ballast = pd.read_sql(query_ballast, engine, params=params)
        st.write(f"Ballast data query successful. Retrieved {len(df_ballast)} rows.")
    except Exception as e:
        st.error(f"Error executing Ballast data query: {str(e)}")
        st.write(f"Full traceback: {traceback.format_exc()}")
        df_ballast = pd.DataFrame()

    try:
        st.write(f"Executing query for Scantling data with parameters: {params}")
        df_scantling = pd.read_sql(query_scantling, engine, params=params)
        st.write(f"Scantling data query successful. Retrieved {len(df_scantling)} rows.")
    except Exception as e:
        st.error(f"Error executing Scantling data query: {str(e)}")
        st.write(f"Full traceback: {traceback.format_exc()}")
        df_scantling = pd.DataFrame()

    if df_scantling.empty:
        try:
            st.write(f"Executing query for Design data with parameters: {params}")
            df_scantling = pd.read_sql(query_design, engine, params=params)
            st.write(f"Design data query successful. Retrieved {len(df_scantling)} rows.")
        except Exception as e:
            st.error(f"Error executing Design data query: {str(e)}")
            st.write(f"Full traceback: {traceback.format_exc()}")
            df_scantling = pd.DataFrame()

    return df_ballast, df_scantling

# ... (rest of the code remains the same)

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
        vessel_names = similar_vessels['vessel_name'].tolist()
        
        st.write("Attempting to fetch performance data...")
        df_ballast, df_scantling = get_vessel_performance_data(engine, vessel_names)
        
        if df_ballast.empty and df_scantling.empty:
            st.write("No performance data available for the selected vessels.")
        else:
            st.write("Data fetched successfully. Ready for verification.")
            
            # Display the dataframes
            if not df_ballast.empty:
                st.write("Ballast Load Type Data:")
                st.write(df_ballast)
            else:
                st.write("No Ballast data available.")
            
            if not df_scantling.empty:
                st.write("Scantling/Design Load Type Data:")
                st.write(df_scantling)
            else:
                st.write("No Scantling/Design data available.")
            st.write("Data fetched successfully. Ready for model training.")

    # Placeholder for model training code
    st.write("Model training functionality is currently hashed out.")

    # Placeholder for output table
    st.write("Creating output table (Speed, Power, Consumption, Displacement)...")
    # Add code to create and display output table here

# Placeholder for future steps and more details
st.sidebar.write("Once the model is trained, we will generate output tables and further analyze results.")
