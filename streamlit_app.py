import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from database import get_db_engine

# Database setup
def get_db_connection():
    return get_db_engine()

# Sidebar for model selection
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
        return pd.read_sql(query, engine, params=params)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()

# Function to get speed, consumption, power data for selected vessels
def get_vessel_performance_data(engine, vessel_names):
    query = """
    SELECT VESSEL_NAME, speed_kts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s
    """
    params = {
        'vessel_names_list': tuple([name.upper() for name in vessel_names])
    }
    try:
        return pd.read_sql(query, engine, params=params)
    except Exception as e:
        st.error(f"Error executing performance data query: {e}")
        return pd.DataFrame()

# Function to train the selected model
def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    elif model_type == "MLP Regressor":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
    
    return model

# Main execution
if st.sidebar.button("Fetch Data and Train Model"):
    engine = get_db_connection()
    similar_vessels = get_similar_vessels(engine, lpp, breadth, depth, deadweight, vessel_type)
    
    if similar_vessels.empty:
        st.write("No vessels found matching the given criteria.")
    else:
        st.write(f"Found {len(similar_vessels)} vessels matching the criteria.")
        st.write("Names of similar vessels:")
        st.write(similar_vessels['vessel_name'].tolist())
        vessel_names = similar_vessels['vessel_name'].tolist()
        
        df_performance = get_vessel_performance_data(engine, vessel_names)
        
        if df_performance.empty:
            st.write("No performance data available for the selected vessels.")
        else:
            # Prepare data for model training
            X = df_performance[['speed_kts', 'displacement']]
            y_power = df_performance['me_power_kw']
            y_consumption = df_performance['me_consumption_mt']
            
            # Train models for power and consumption
            model_power = train_model(X, y_power, selected_model)
            model_consumption = train_model(X, y_consumption, selected_model)
            
            # Create output table
            st.subheader("Output Table (Predictions)")
            
            # Set speed range based on vessel type
            if vessel_type == "CONTAINER":
                output_speeds = range(10, 23)  # 10 to 22 knots
            else:
                output_speeds = range(8, 16)  # 8 to 15 knots
            
            ballast_displacement = df_performance['displacement'].min()
            laden_displacement = df_performance['displacement'].max()
            
            output_data = []
            
            for speed in output_speeds:
                for disp in [ballast_displacement, laden_displacement]:
                    if selected_model == "Linear Regression with Polynomial Features":
                        power = model_power.predict(PolynomialFeatures(degree=2).fit_transform([[speed, disp]]))[0]
                        consumption = model_consumption.predict(PolynomialFeatures(degree=2).fit_transform([[speed, disp]]))[0]
                    else:
                        power = model_power.predict([[speed, disp]])[0]
                        consumption = model_consumption.predict([[speed, disp]])[0]
                    output_data.append({
                        'Speed (kts)': speed,
                        'Displacement': 'Ballast' if disp == ballast_displacement else 'Laden',
                        'Predicted Power (kW)': round(power, 2),
                        'Predicted Consumption (mt/day)': round(consumption * 24, 2)  # Assuming consumption is per hour, converting to per day
                    })
            
            output_df = pd.DataFrame(output_data)
            st.dataframe(output_df)

st.sidebar.write("Once the models are trained, you can analyze the predictions in the output table.")
