import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from database import get_db_engine
import sqlalchemy

def get_db_connection():
    return get_db_engine()

st.sidebar.header("Vessel Details and Model Selection")
lpp = st.sidebar.number_input("Lpp (m)", min_value=50.0, max_value=400.0, step=0.1)
breadth = st.sidebar.number_input("Breadth (m)", min_value=10.0, max_value=100.0, step=0.1)
depth = st.sidebar.number_input("Depth (m)", min_value=5.0, max_value=50.0, step=0.1)
deadweight = st.sidebar.number_input("Deadweight (tons)", min_value=1000, max_value=500000, step=100)
mcr = st.sidebar.number_input("MCR of Main Engine (kW)", min_value=500, max_value=100000, step=100)
vessel_type = st.sidebar.selectbox("Vessel Type", ["BULK CARRIER", "CONTAINER", "OIL TANKER"])

# Model selection
model_options = ["Linear Regression with Polynomial Features", "Random Forest", "MLP Regressor"]
selected_model = st.sidebar.selectbox("Select a model to train:", model_options)

def get_hull_data(engine, vessel_type, limit=None):
    query = """
    SELECT length_between_perpendiculars_m as lpp, breadth_moduled_m as breadth, 
           depth, deadweight, me_1_mcr_kw as mcr, imo, vessel_name
    FROM hull_particulars
    WHERE vessel_type = %(vessel_type)s
    """
    if limit:
        query += f" ORDER BY RANDOM() LIMIT {limit}"
    df = pd.read_sql(query, engine, params={'vessel_type': vessel_type})
    return df.dropna()

def get_performance_data(engine, imos):
    imos = [int(imo) for imo in imos]
    query = """
    SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo, load_type
    FROM vessel_performance_model_data
    WHERE vessel_imo IN %(imos)s
    """
    df = pd.read_sql(query, engine, params={'imos': tuple(imos)})
    return df.dropna()

def separate_data(df):
    ballast_df = df[df['load_type'] == 'Ballast']
    laden_df = df[df['load_type'].isin(['Scantling', 'Design'])]
    
    if 'Scantling' in laden_df['load_type'].values:
        laden_df = laden_df[laden_df['load_type'] == 'Scantling']
    else:
        laden_df = laden_df[laden_df['load_type'] == 'Design']
    
    return ballast_df, laden_df

def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_train_scaled)
        model = LinearRegression()
        model.fit(X_poly, y_train)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
    elif model_type == "MLP Regressor":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_performance(model, scaler, input_data, model_type):
    input_scaled = scaler.transform(input_data)
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        input_poly = poly.fit_transform(input_scaled)
        prediction = model.predict(input_poly)
    else:
        prediction = model.predict(input_scaled)
    return np.maximum(0, prediction)

def calculate_percentage_difference(actual, predicted):
    return np.abs((actual - predicted) / actual) * 100

def run_test():
    try:
        engine = get_db_connection()
        
        # Get 10 random bulk carriers
        test_vessels = get_hull_data(engine, "BULK CARRIER", limit=10)
        
        # Get performance data for test vessels
        performance_data = get_performance_data(engine, test_vessels['imo'].unique())
        
        # Combine hull and performance data
        combined_data = pd.merge(test_vessels, performance_data, left_on='imo', right_on='vessel_imo')
        
        # Separate data into ballast and laden conditions
        ballast_df, laden_df = separate_data(combined_data)
        
        input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
        
        results = {}
        
        for model_type in model_options:
            st.write(f"Testing {model_type}")
            
            # Train models
            ballast_power_model, ballast_power_scaler = train_model(ballast_df[input_columns], ballast_df['me_power_kw'], model_type)
            ballast_consumption_model, ballast_consumption_scaler = train_model(ballast_df[input_columns], ballast_df['me_consumption_mt'], model_type)
            laden_power_model, laden_power_scaler = train_model(laden_df[input_columns], laden_df['me_power_kw'], model_type)
            laden_consumption_model, laden_consumption_scaler = train_model(laden_df[input_columns], laden_df['me_consumption_mt'], model_type)
            
            # Initialize results dictionary for this model
            results[model_type] = {
                'ballast_power': [], 'ballast_consumption': [],
                'laden_power': [], 'laden_consumption': []
            }
            
            # Generate predictions and calculate differences
            for speed in range(8, 16):
                ballast_power_diff = []
                ballast_consumption_diff = []
                laden_power_diff = []
                laden_consumption_diff = []
                
                for _, vessel in test_vessels.iterrows():
                    input_data = pd.DataFrame([[vessel['lpp'], vessel['breadth'], vessel['depth'], 
                                                vessel['deadweight'], vessel['mcr'], speed]], 
                                              columns=input_columns)
                    
                    # Ballast predictions
                    ballast_power_pred = predict_performance(ballast_power_model, ballast_power_scaler, input_data, model_type)[0]
                    ballast_consumption_pred = predict_performance(ballast_consumption_model, ballast_consumption_scaler, input_data, model_type)[0]
                    
                    # Laden predictions
                    laden_power_pred = predict_performance(laden_power_model, laden_power_scaler, input_data, model_type)[0]
                    laden_consumption_pred = predict_performance(laden_consumption_model, laden_consumption_scaler, input_data, model_type)[0]
                    
                    # Get actual values
                    ballast_actual = ballast_df[(ballast_df['imo'] == vessel['imo']) & (ballast_df['speed_kts'].round() == speed)]
                    laden_actual = laden_df[(laden_df['imo'] == vessel['imo']) & (laden_df['speed_kts'].round() == speed)]
                    
                    if not ballast_actual.empty:
                        ballast_power_diff.append(calculate_percentage_difference(ballast_actual['me_power_kw'].values[0], ballast_power_pred))
                        ballast_consumption_diff.append(calculate_percentage_difference(ballast_actual['me_consumption_mt'].values[0], ballast_consumption_pred))
                    
                    if not laden_actual.empty:
                        laden_power_diff.append(calculate_percentage_difference(laden_actual['me_power_kw'].values[0], laden_power_pred))
                        laden_consumption_diff.append(calculate_percentage_difference(laden_actual['me_consumption_mt'].values[0], laden_consumption_pred))
                
                # Calculate average differences
                results[model_type]['ballast_power'].append(np.mean(ballast_power_diff) if ballast_power_diff else np.nan)
                results[model_type]['ballast_consumption'].append(np.mean(ballast_consumption_diff) if ballast_consumption_diff else np.nan)
                results[model_type]['laden_power'].append(np.mean(laden_power_diff) if laden_power_diff else np.nan)
                results[model_type]['laden_consumption'].append(np.mean(laden_consumption_diff) if laden_consumption_diff else np.nan)
            
            # Create and display results table
            results_df = pd.DataFrame({
                'Speed (kts)': range(8, 16),
                'Ballast Power % Diff': results[model_type]['ballast_power'],
                'Ballast Consumption % Diff': results[model_type]['ballast_consumption'],
                'Laden Power % Diff': results[model_type]['laden_power'],
                'Laden Consumption % Diff': results[model_type]['laden_consumption']
            }).set_index('Speed (kts)')
            
            st.dataframe(results_df.style.format("{:.2f}"))
        
        return results
    
    except Exception as e:
        st.error(f"An error occurred during testing: {str(e)}")
        return None

# Main execution
col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Predictions"):
        try:
            engine = get_db_connection()
            hull_data = get_hull_data(engine, vessel_type)
            performance_data = get_performance_data(engine, hull_data['imo'].unique())
            
            # Merge hull data with performance data
            combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo').dropna()
            
            # Separate data into ballast and laden conditions
            ballast_df, laden_df = separate_data(combined_data)
            
            # Display the separated datasets
            st.subheader("Separated Datasets")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Ballast Condition Data")
                st.dataframe(ballast_df[['imo', 'vessel_name', 'lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts', 'me_power_kw', 'me_consumption_mt']])
            
            with col2:
                st.write("Laden Condition Data")
                st.dataframe(laden_df[['imo', 'vessel_name', 'lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts', 'me_power_kw', 'me_consumption_mt']])
            
            input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
            
            ballast_power_model, ballast_power_scaler = train_model(ballast_df[input_columns], ballast_df['me_power_kw'], selected_model)
            ballast_consumption_model, ballast_consumption_scaler = train_model(ballast_df[input_columns], ballast_df['me_consumption_mt'], selected_model)
            laden_power_model, laden_power_scaler = train_model(laden_df[input_columns], laden_df['me_power_kw'], selected_model)
            laden_consumption_model, laden_consumption_scaler = train_model(laden_df[input_columns], laden_df['me_consumption_mt'], selected_model)
            
            speed_range = range(10, 23) if vessel_type == "CONTAINER" else range(8, 16)
            
            ballast_predictions = []
            laden_predictions = []
            
            for speed in speed_range:
                input_data = pd.DataFrame([[lpp, breadth, depth, deadweight, mcr, speed]], columns=input_columns)
                
                ballast_power = predict_performance(ballast_power_model, ballast_power_scaler, input_data, selected_model)[0]
                ballast_consumption = predict_performance(ballast_consumption_model, ballast_consumption_scaler, input_data, selected_model)[0]
                laden_power = predict_performance(laden_power_model, laden_power_scaler, input_data, selected_model)[0]
                laden_consumption = predict_performance(laden_consumption_model, laden_consumption_scaler, input_data, selected_model)[0]
                
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

with col2:
    if st.button("Run Test"):
        test_results = run_test()
        if test_results:
            st.subheader("Test Results Summary")
            for model, results in test_results.items():
                st.write(f"Model: {model}")
                st.write(pd.DataFrame(results).mean())
                st.write("---")
