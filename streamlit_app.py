import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from database import get_db_engine
import traceback

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
        st.write(f"Executing query: {query}")
        st.write(f"With parameters: {params}")
        return pd.read_sql(query, engine, params=params)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        st.write(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()

# Function to get speed, consumption, power data for selected vessels
def get_vessel_performance_data(engine, vessel_names):
    query_ballast = """
    SELECT VESSEL_NAME, speed_kts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s AND load_type = 'Ballast'
    """
    query_scantling = """
    SELECT VESSEL_NAME, speed_kts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s AND load_type = 'Scantling'
    """
    query_design = """
    SELECT VESSEL_NAME, speed_kts, me_power_kw, me_consumption_mt, displacement
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

# Function to train the selected model
def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
        y_pred = model.predict(poly.transform(X_test))
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "MLP Regressor":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

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
        
        st.write("Attempting to fetch performance data...")
        df_ballast, df_scantling = get_vessel_performance_data(engine, vessel_names)
        
        if df_ballast.empty and df_scantling.empty:
            st.write("No performance data available for the selected vessels.")
        else:
            st.write("Data fetched successfully. Ready for model training.")
            
            # Combine ballast and scantling data
            df_combined = pd.concat([df_ballast, df_scantling], ignore_index=True)
            
            # Prepare data for model training
            X = df_combined[['speed_kts', 'displacement']]
            y_power = df_combined['me_power_kw']
            y_consumption = df_combined['me_consumption_mt']
            
            # Train models for power and consumption
            model_power, mse_power, r2_power = train_model(X, y_power, selected_model)
            model_consumption, mse_consumption, r2_consumption = train_model(X, y_consumption, selected_model)
            
            st.write(f"Model: {selected_model}")
            st.write("Power Prediction Model:")
            st.write(f"Mean Squared Error: {mse_power}")
            st.write(f"R-squared Score: {r2_power}")
            st.write("Consumption Prediction Model:")
            st.write(f"Mean Squared Error: {mse_consumption}")
            st.write(f"R-squared Score: {r2_consumption}")
            
            # Create output table
            st.subheader("Output Table (Sample Predictions)")
            output_speeds = np.linspace(X['speed_kts'].min(), X['speed_kts'].max(), 10)
            output_displacements = np.linspace(X['displacement'].min(), X['displacement'].max(), 10)
            output_data = []
            
            for speed in output_speeds:
                for disp in output_displacements:
                    if selected_model == "Linear Regression with Polynomial Features":
                        power = model_power.predict(PolynomialFeatures(degree=2).fit_transform([[speed, disp]]))[0]
                        consumption = model_consumption.predict(PolynomialFeatures(degree=2).fit_transform([[speed, disp]]))[0]
                    else:
                        power = model_power.predict([[speed, disp]])[0]
                        consumption = model_consumption.predict([[speed, disp]])[0]
                    output_data.append({
                        'Speed (kts)': speed,
                        'Displacement': disp,
                        'Predicted Power (kW)': power,
                        'Predicted Consumption (mt)': consumption
                    })
            
            output_df = pd.DataFrame(output_data)
            st.dataframe(output_df)

st.sidebar.write("Once the models are trained, you can analyze the results and make predictions for both power and consumption.")
