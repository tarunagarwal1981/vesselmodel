import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from database import get_db_engine
import sqlalchemy

def get_db_connection():
    return get_db_engine()

st.sidebar.header("Vessel Details")
lpp = st.sidebar.number_input("Lpp (m)", min_value=50.0, max_value=400.0, step=0.1)
breadth = st.sidebar.number_input("Breadth (m)", min_value=10.0, max_value=100.0, step=0.1)
depth = st.sidebar.number_input("Depth (m)", min_value=5.0, max_value=50.0, step=0.1)
deadweight = st.sidebar.number_input("Deadweight (tons)", min_value=1000, max_value=500000, step=100)
mcr = st.sidebar.number_input("MCR of Main Engine (kW)", min_value=500, max_value=100000, step=100)
vessel_type = st.sidebar.selectbox("Vessel Type", ["BULK CARRIER", "CONTAINER", "OIL TANKER"])

def get_hull_data(engine, vessel_type):
    query = """
    SELECT length_between_perpendiculars_m as lpp, breadth_moduled_m as breadth, 
           depth, deadweight, me_1_mcr_kw as mcr, imo
    FROM hull_particulars
    WHERE vessel_type = %(vessel_type)s
    """
    df = pd.read_sql(query, engine, params={'vessel_type': vessel_type})
    return df.dropna()

def get_performance_data(engine, imos):
    # Convert numpy.int64 to regular Python int
    imos = [int(imo) for imo in imos]
    query = """
    SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo
    FROM vessel_performance_model_data
    WHERE vessel_imo IN %(imos)s
    """
    df = pd.read_sql(query, engine, params={'imos': tuple(imos)})
    return df.dropna()

def separate_data(df):
    df_sorted = df.sort_values('deadweight')
    split_index = len(df) // 2
    return df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def predict_performance(model, scaler, input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return max(0, prediction[0])

if st.sidebar.button("Generate Predictions"):
    try:
        engine = get_db_connection()
        hull_data = get_hull_data(engine, vessel_type)
        performance_data = get_performance_data(engine, hull_data['imo'].unique())
        combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo').dropna()
        
        ballast_df, laden_df = separate_data(combined_data)
        
        input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
        
        ballast_power_model, ballast_power_scaler = train_model(ballast_df[input_columns], ballast_df['me_power_kw'])
        ballast_consumption_model, ballast_consumption_scaler = train_model(ballast_df[input_columns], ballast_df['me_consumption_mt'])
        laden_power_model, laden_power_scaler = train_model(laden_df[input_columns], laden_df['me_power_kw'])
        laden_consumption_model, laden_consumption_scaler = train_model(laden_df[input_columns], laden_df['me_consumption_mt'])
        
        speed_range = range(10, 23) if vessel_type == "CONTAINER" else range(8, 16)
        
        ballast_predictions = []
        laden_predictions = []
        
        for speed in speed_range:
            input_data = pd.DataFrame([[lpp, breadth, depth, deadweight, mcr, speed]], columns=input_columns)
            
            ballast_power = predict_performance(ballast_power_model, ballast_power_scaler, input_data)
            ballast_consumption = predict_performance(ballast_consumption_model, ballast_consumption_scaler, input_data)
            laden_power = predict_performance(laden_power_model, laden_power_scaler, input_data)
            laden_consumption = predict_performance(laden_consumption_model, laden_consumption_scaler, input_data)
            
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
        
        st.subheader("Predicted Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Ballast Condition")
            st.dataframe(pd.DataFrame(ballast_predictions).set_index('Speed (kts)'))
        
        with col2:
            st.write("Laden Condition")
            st.dataframe(pd.DataFrame(laden_predictions).set_index('Speed (kts)'))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
