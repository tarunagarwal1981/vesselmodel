import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from database import get_db_engine
import sqlalchemy

# Database setup
def get_db_connection():
    try:
        return get_db_engine()
    except Exception as e:
        st.error(f"Failed to connect to the database: {str(e)}")
        return None

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
    try:
        query = """
        SELECT length_between_perpendiculars_m as lpp, breadth_moduled_m as breadth, 
               depth, deadweight, me_1_mcr_kw as mcr, imo
        FROM hull_particulars
        WHERE vessel_type = %(vessel_type)s
        """
        st.write(f"Debug - SQL Query: {query}")
        st.write(f"Debug - vessel_type: {vessel_type}")
        
        df = pd.read_sql(query, engine, params={'vessel_type': vessel_type})
        st.write(f"Debug - DataFrame columns: {df.columns}")
        st.write(f"Debug - DataFrame shape: {df.shape}")
        return df
    except sqlalchemy.exc.SQLAlchemyError as e:
        st.error(f"Database error in get_hull_data: {str(e)}")
        return pd.DataFrame()

def get_performance_data(engine, imos):
    try:
        imos = [int(imo) for imo in imos]
        query = """
        SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo
        FROM vessel_performance_model_data
        WHERE vessel_imo IN %(imos)s
        """
        st.write(f"Debug - Performance SQL Query: {query}")
        st.write(f"Debug - Number of IMOs: {len(imos)}")
        
        df = pd.read_sql(query, engine, params={'imos': tuple(imos)})
        st.write(f"Debug - Performance DataFrame columns: {df.columns}")
        st.write(f"Debug - Performance DataFrame shape: {df.shape}")
        return df
    except sqlalchemy.exc.SQLAlchemyError as e:
        st.error(f"Database error in get_performance_data: {str(e)}")
        return pd.DataFrame()

def separate_data(df):
    try:
        # Since we don't have loading_condition, we'll use a simple split based on deadweight
        df_sorted = df.sort_values('deadweight')
        split_index = len(df) // 2
        ballast = df_sorted.iloc[:split_index]
        laden = df_sorted.iloc[split_index:]
        
        st.write(f"Debug - Ballast DataFrame shape: {ballast.shape}")
        st.write(f"Debug - Laden DataFrame shape: {laden.shape}")
        return ballast, laden
    except Exception as e:
        st.error(f"Error in separate_data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def train_model(X, y, model_type):
    try:
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
    except Exception as e:
        st.error(f"Error in train_model: {str(e)}")
        return None

def predict_performance(model, input_data, model_type):
    try:
        if model_type == "Linear Regression with Polynomial Features":
            poly = PolynomialFeatures(degree=2)
            input_poly = poly.fit_transform(input_data)
            return model.predict(input_poly)
        else:
            return model.predict(input_data)
    except Exception as e:
        st.error(f"Error in predict_performance: {str(e)}")
        return None

# Main execution
if st.sidebar.button("Fetch Data and Train Model"):
    try:
        engine = get_db_connection()
        if engine is None:
            st.error("Failed to establish database connection")
        else:
            st.write("Debug - Database connection successful")
            
            # Get hull data
            hull_data = get_hull_data(engine, vessel_type)
            
            if hull_data.empty:
                st.error("No hull data retrieved from the database. Please check your input parameters.")
            else:
                # Get performance data
                performance_data = get_performance_data(engine, hull_data['imo'].unique())
                
                if performance_data.empty:
                    st.error("No performance data retrieved from the database.")
                else:
                    # Combine hull and performance data
                    combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo')
                    
                    # Separate data
                    ballast_df, laden_df = separate_data(combined_data)
                    
                    st.write(f"Debug - Combined Ballast DataFrame shape: {ballast_df.shape}")
                    st.write(f"Debug - Combined Laden DataFrame shape: {laden_df.shape}")
                    
                    # Train models
                    input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
                    
                    ballast_power_model = train_model(ballast_df[input_columns], ballast_df['me_power_kw'], selected_model)
                    ballast_consumption_model = train_model(ballast_df[input_columns], ballast_df['me_consumption_mt'], selected_model)
                    
                    laden_power_model = train_model(laden_df[input_columns], laden_df['me_power_kw'], selected_model)
                    laden_consumption_model = train_model(laden_df[input_columns], laden_df['me_consumption_mt'], selected_model)
                    
                    if all([ballast_power_model, ballast_consumption_model, laden_power_model, laden_consumption_model]):
                        # Generate predictions
                        if vessel_type == "CONTAINER":
                            speed_range = range(10, 23)
                        else:
                            speed_range = range(8, 16)
                        
                        ballast_predictions = []
                        laden_predictions = []
                        
                        for speed in speed_range:
                            input_data = np.array([[lpp, breadth, depth, deadweight, mcr, speed]])
                            
                            ballast_power = predict_performance(ballast_power_model, input_data, selected_model)
                            ballast_consumption = predict_performance(ballast_consumption_model, input_data, selected_model)
                            
                            laden_power = predict_performance(laden_power_model, input_data, selected_model)
                            laden_consumption = predict_performance(laden_consumption_model, input_data, selected_model)
                            
                            if all([ballast_power is not None, ballast_consumption is not None, 
                                    laden_power is not None, laden_consumption is not None]):
                                ballast_predictions.append({
                                    'Speed (kts)': speed,
                                    'Power (kW)': round(ballast_power[0], 2),
                                    'Consumption (mt/day)': round(ballast_consumption[0], 2)
                                })
                                
                                laden_predictions.append({
                                    'Speed (kts)': speed,
                                    'Power (kW)': round(laden_power[0], 2),
                                    'Consumption (mt/day)': round(laden_consumption[0], 2)
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
                    else:
                        st.error("Failed to train one or more models. Please check the input data.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

st.sidebar.write("Once the models are trained, you can analyze the predictions in the output tables.")
