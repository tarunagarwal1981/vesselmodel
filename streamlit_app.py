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

# Sidebar for model selection and vessel details
st.sidebar.header("Model Selection and Vessel Details")
model_options = ["Linear Regression with Polynomial Features", "Random Forest", "MLP Regressor"]
selected_model = st.sidebar.selectbox("Select a model to train:", model_options)

# User Inputs for Vessel Particulars
lpp = st.sidebar.number_input("Lpp (m)", min_value=50.0, max_value=400.0, step=0.1)
breadth = st.sidebar.number_input("Breadth (m)", min_value=10.0, max_value=100.0, step=0.1)
depth = st.sidebar.number_input("Depth (m)", min_value=5.0, max_value=50.0, step=0.1)
deadweight = st.sidebar.number_input("Deadweight (tons)", min_value=1000, max_value=500000, step=100)
mcr = st.sidebar.number_input("MCR of Main Engine (kW)", min_value=500, max_value=100000, step=100)
vessel_type = st.sidebar.selectbox("Vessel Type", ["BULK CARRIER", "CONTAINER", "OIL TANKER"])

def get_hull_data(engine, vessel_type):
    query = """
    SELECT lpp, breadth, depth, deadweight, mcr, load_type, imo
    FROM hull_particulars
    WHERE vessel_type = %(vessel_type)s
    """
    return pd.read_sql(query, engine, params={'vessel_type': vessel_type})

def get_performance_data(engine, imos):
    query = """
    SELECT speed_kts, me_consumption_mt, me_power_kw, load_type, vessel_imo
    FROM vessel_performance_model_data
    WHERE vessel_imo IN %(imos)s
    """
    return pd.read_sql(query, engine, params={'imos': tuple(imos)})

def separate_data(df, load_type_column):
    ballast = df[df[load_type_column] == 'Ballast']
    laden = df[(df[load_type_column] == 'Scantling') | (df[load_type_column] == 'Design')]
    laden = laden.drop_duplicates(subset=['imo' if 'imo' in df.columns else 'vessel_imo'], keep='first')
    return ballast, laden

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

def predict_performance(model, input_data, model_type):
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        input_poly = poly.fit_transform(input_data)
        return model.predict(input_poly)
    else:
        return model.predict(input_data)

# Main execution
if st.sidebar.button("Fetch Data and Train Model"):
    engine = get_db_connection()
    
    # Get hull data
    hull_data = get_hull_data(engine, vessel_type)
    
    # Separate hull data
    hull_ballast, hull_laden = separate_data(hull_data, 'load_type')
    
    # Get performance data
    performance_data = get_performance_data(engine, hull_data['imo'].unique())
    
    # Separate performance data
    perf_ballast, perf_laden = separate_data(performance_data, 'load_type')
    
    # Combine data
    ballast_df = pd.merge(hull_ballast, perf_ballast, left_on='imo', right_on='vessel_imo')
    laden_df = pd.merge(hull_laden, perf_laden, left_on='imo', right_on='vessel_imo')
    
    # Train models
    input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
    
    ballast_power_model = train_model(ballast_df[input_columns], ballast_df['me_power_kw'], selected_model)
    ballast_consumption_model = train_model(ballast_df[input_columns], ballast_df['me_consumption_mt'], selected_model)
    
    laden_power_model = train_model(laden_df[input_columns], laden_df['me_power_kw'], selected_model)
    laden_consumption_model = train_model(laden_df[input_columns], laden_df['me_consumption_mt'], selected_model)
    
    # Generate predictions
    if vessel_type == "CONTAINER":
        speed_range = range(10, 23)
    else:
        speed_range = range(8, 16)
    
    ballast_predictions = []
    laden_predictions = []
    
    for speed in speed_range:
        input_data = np.array([[lpp, breadth, depth, deadweight, mcr, speed]])
        
        ballast_power = predict_performance(ballast_power_model, input_data, selected_model)[0]
        ballast_consumption = predict_performance(ballast_consumption_model, input_data, selected_model)[0]
        
        laden_power = predict_performance(laden_power_model, input_data, selected_model)[0]
        laden_consumption = predict_performance(laden_consumption_model, input_data, selected_model)[0]
        
        ballast_predictions.append({
            'Speed (kts)': speed,
            'Power (kW)': round(ballast_power, 2),
            'Consumption (mt/day)': round(ballast_consumption, 2)
        })
        
        laden_predictions.append({
            'Speed (kts)': speed,
            'Power (kW)': round(laden_power, 2),
            'Consumption (mt/day)': round(laden_consumption, 2)
        })
    
    # Display results
    st.subheader("Predicted Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Ballast Condition")
        st.dataframe(pd.DataFrame(ballast_predictions).set_index('Speed (kts)'))
    
    with col2:
        st.write("Laden Condition")
        st.dataframe(pd.DataFrame(laden_predictions).set_index('Speed (kts)'))

st.sidebar.write("Once the models are trained, you can analyze the predictions in the output tables.")
