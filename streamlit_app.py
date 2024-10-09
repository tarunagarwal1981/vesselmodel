import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from database import get_db_engine

# Database setup
def get_db_connection():
    return get_db_engine()

# Sidebar for model selection and vessel details
st.sidebar.header("Model Selection and Vessel Details")
model_options = ["Linear Regression with Polynomial Features", "Random Forest", "MLP Regressor"]
selected_model = st.sidebar.selectbox("Select a model to train:", model_options)

vessel_type = st.sidebar.selectbox("Vessel Type", ["BULK CARRIER", "CONTAINER", "OIL TANKER"])

# Function to get vessel data from hull particulars
def get_hull_data(engine, vessel_type):
    query = """
    SELECT lpp, breadth, depth, deadweight, mcr, load_type, imo
    FROM hull_particulars
    WHERE vessel_type = %s
    """
    return pd.read_sql(query, engine, params=(vessel_type,))

# Function to get performance data
def get_performance_data(engine, imos):
    query = """
    SELECT speed_kts, me_consumption_mt, me_power_kw, load_type, vessel_imo
    FROM vessel_performance_model_data
    WHERE vessel_imo IN %s
    """
    return pd.read_sql(query, engine, params=(tuple(imos),))

# Function to separate data by load type
def separate_by_load_type(df):
    ballast = df[df['load_type'] == 'Ballast']
    laden = df[(df['load_type'] == 'Scantling') | 
               ((df['load_type'] == 'Design') & (~df['imo'].isin(df[df['load_type'] == 'Scantling']['imo'])))]
    return ballast, laden

# Function to train model
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
    
    return model, poly if model_type == "Linear Regression with Polynomial Features" else None

# Main execution
if st.sidebar.button("Fetch Data and Train Model"):
    engine = get_db_connection()
    
    # Get hull data
    hull_data = get_hull_data(engine, vessel_type)
    
    # Separate hull data
    hull_ballast, hull_laden = separate_by_load_type(hull_data)
    
    # Get performance data
    performance_data = get_performance_data(engine, hull_data['imo'].unique())
    
    # Separate performance data
    perf_ballast, perf_laden = separate_by_load_type(performance_data)
    
    # Combine data
    ballast_df = pd.merge(hull_ballast, perf_ballast, left_on='imo', right_on='vessel_imo')
    laden_df = pd.merge(hull_laden, perf_laden, left_on='imo', right_on='vessel_imo')
    
    # Train models
    features = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
    
    ballast_power_model, ballast_power_poly = train_model(ballast_df[features], ballast_df['me_power_kw'], selected_model)
    ballast_consumption_model, ballast_consumption_poly = train_model(ballast_df[features], ballast_df['me_consumption_mt'], selected_model)
    
    laden_power_model, laden_power_poly = train_model(laden_df[features], laden_df['me_power_kw'], selected_model)
    laden_consumption_model, laden_consumption_poly = train_model(laden_df[features], laden_df['me_consumption_mt'], selected_model)
    
    # Generate predictions
    if vessel_type == "CONTAINER":
        speeds = range(10, 23)
    else:
        speeds = range(8, 16)
    
    ballast_predictions = []
    laden_predictions = []
    
    for speed in speeds:
        input_data = np.array([hull_ballast.iloc[0][['lpp', 'breadth', 'depth', 'deadweight', 'mcr']].tolist() + [speed]])
        
        if selected_model == "Linear Regression with Polynomial Features":
            ballast_power = ballast_power_model.predict(ballast_power_poly.transform(input_data))[0]
            ballast_consumption = ballast_consumption_model.predict(ballast_consumption_poly.transform(input_data))[0]
            laden_power = laden_power_model.predict(laden_power_poly.transform(input_data))[0]
            laden_consumption = laden_consumption_model.predict(laden_consumption_poly.transform(input_data))[0]
        else:
            ballast_power = ballast_power_model.predict(input_data)[0]
            ballast_consumption = ballast_consumption_model.predict(input_data)[0]
            laden_power = laden_power_model.predict(input_data)[0]
            laden_consumption = laden_consumption_model.predict(input_data)[0]
        
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
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Ballast Condition")
        st.dataframe(pd.DataFrame(ballast_predictions).set_index('Speed (kts)'))
    
    with col2:
        st.write("Laden Condition")
        st.dataframe(pd.DataFrame(laden_predictions).set_index('Speed (kts)'))

st.sidebar.write("Once the models are trained, you can analyze the predictions in the output tables.")
