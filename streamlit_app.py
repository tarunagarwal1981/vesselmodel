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
from test import run_all_tests

def get_db_connection():
    engine = get_db_engine()
    
    # Configure engine with timeouts
    if hasattr(engine, 'pool'):
        engine.pool._timeout = 300  # 5 minutes
        engine.pool._recycle = 3600  # 1 hour
    
    return engine

st.set_page_config(page_title="Vessel Performance Prediction", layout="wide")

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

def get_hull_data(engine, vessel_type):
    # First, try a simpler query to check if data exists
    try:
        # Simple count query first
        count_query = sqlalchemy.text("SELECT COUNT(*) FROM hull_particulars WHERE vessel_type = :vessel_type")
        with engine.connect() as conn:
            # Set timeout for this session
            conn.execute(sqlalchemy.text("SET statement_timeout = '60s'"))
            result = conn.execute(count_query, {'vessel_type': vessel_type})
            count = result.scalar()
            st.write(f"Found {count} total records for {vessel_type}")
        
        if count == 0:
            return pd.DataFrame()
        
        # If count is reasonable, proceed with full query
        query = sqlalchemy.text("""
        SELECT 
            "length_between_perpendiculars_m" as lpp, 
            "breadth_moduled_m" as breadth, 
            "depth" as depth, 
            deadweight, 
            "me_1_mcr_kw" as mcr, 
            "imo" as imo, 
            "vessel_name" as vessel_name
        FROM hull_particulars
        WHERE vessel_type = :vessel_type
        LIMIT 1000
        """)
        
        # Use connection with timeout
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SET statement_timeout = '120s'"))
            result = conn.execute(query, {'vessel_type': vessel_type})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            st.write("No data returned from query")
            return df
        
        st.write(f"Retrieved {len(df)} records")
        
        # Convert string columns to numeric where needed
        numeric_columns = ['lpp', 'breadth', 'depth', 'mcr', 'deadweight']
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure IMO is string for consistent merging
        if 'imo' in df.columns:
            df['imo'] = df['imo'].astype(str).str.strip()
        
        # Filter out rows with null values in critical columns
        initial_count = len(df)
        df_clean = df.dropna(subset=['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'imo'])
        st.write(f"After cleaning: {len(df_clean)} usable records (removed {initial_count - len(df_clean)} with missing data)")
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error fetching hull data: {str(e)}")
        # Fallback: try without WHERE clause to test basic connectivity
        try:
            st.write("Trying basic connectivity test...")
            simple_query = sqlalchemy.text("""
            SELECT 
                "Length_between_perpendiculars_m" as lpp, 
                "Breadth_Moduled_m" as breadth, 
                "Depth" as depth, 
                deadweight, 
                "ME_1_MCR_kW" as mcr, 
                "IMO" as imo, 
                "Vessel_Name" as vessel_name,
                vessel_type
            FROM hull_particulars
            LIMIT 10
            """)
            
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SET statement_timeout = '60s'"))
                result = conn.execute(simple_query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            st.write("Sample data from table:")
            st.dataframe(df)
            
            # Filter by vessel type in pandas
            df_filtered = df[df['vessel_type'] == vessel_type] if 'vessel_type' in df.columns else pd.DataFrame()
            
            if not df_filtered.empty:
                # Convert and clean
                numeric_columns = ['lpp', 'breadth', 'depth', 'mcr', 'deadweight']
                for col in numeric_columns:
                    if col in df_filtered.columns and df_filtered[col].dtype == 'object':
                        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                
                return df_filtered.dropna(subset=['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'imo'])
            else:
                st.write(f"No records found for vessel type: {vessel_type}")
                st.write(f"Available vessel types: {df['vessel_type'].unique().tolist() if 'vessel_type' in df.columns else 'Unknown'}")
                return pd.DataFrame()
            
        except Exception as e2:
            st.error(f"Even basic connectivity test failed: {str(e2)}")
            return pd.DataFrame()

def get_performance_data(engine, imos):
    if len(imos) == 0:
        return pd.DataFrame()
        
    imos = [str(imo) for imo in imos]  # Convert to string to match IMO format
    
    try:
        # Limit the number of IMOs to prevent timeout
        limited_imos = imos[:50]  # Process max 50 vessels at a time
        
        query = sqlalchemy.text("""
        SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo, load_type
        FROM vessel_performance_model_data
        WHERE vessel_imo::text = ANY(:imos)
        LIMIT 5000
        """)
        
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SET statement_timeout = '120s'"))
            result = conn.execute(query, {'imos': limited_imos})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Ensure vessel_imo is string for consistent merging
        if not df.empty and 'vessel_imo' in df.columns:
            df['vessel_imo'] = df['vessel_imo'].astype(str).str.strip()
        
        return df.dropna() if not df.empty else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching performance data: {str(e)}")
        # Try a simpler approach
        try:
            st.write("Trying simplified performance data query...")
            simple_query = sqlalchemy.text("""
            SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo, load_type
            FROM vessel_performance_model_data
            LIMIT 1000
            """)
            
            with engine.connect() as conn:
                result = conn.execute(simple_query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                # Filter in pandas
                df_filtered = df[df['vessel_imo'].astype(str).isin(imos)]
                return df_filtered.dropna()
            
        except Exception as e2:
            st.error(f"Performance data fallback also failed: {str(e2)}")
        
        return pd.DataFrame()

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

# Main execution
if st.sidebar.button("Generate Predictions"):
    try:
        engine = get_db_connection()
        
        # Get hull data
        st.write("Fetching hull data...")
        hull_data = get_hull_data(engine, vessel_type)
        
        if hull_data.empty:
            st.error(f"No hull data found for vessel type: {vessel_type}")
            st.stop()
        
        st.write(f"Found {len(hull_data)} vessels of type {vessel_type}")
        
        # Get performance data
        st.write("Fetching performance data...")
        performance_data = get_performance_data(engine, hull_data['imo'].unique())
        
        if performance_data.empty:
            st.error("No performance data found for the selected vessels")
            st.stop()
        
        # Fix data type mismatch before merging
        st.write("Preparing data for merge...")
        
        # Convert IMO columns to string for consistent merging
        hull_data['imo'] = hull_data['imo'].astype(str)
        performance_data['vessel_imo'] = performance_data['vessel_imo'].astype(str)
        
        # Display data types for debugging
        st.write(f"Hull data IMO type: {hull_data['imo'].dtype}")
        st.write(f"Performance data vessel_imo type: {performance_data['vessel_imo'].dtype}")
        st.write(f"Sample hull IMOs: {hull_data['imo'].head().tolist()}")
        st.write(f"Sample performance IMOs: {performance_data['vessel_imo'].head().tolist()}")
        
        # Merge hull data with performance data
        combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo').dropna()
        
        if combined_data.empty:
            st.error("No matching data found between hull particulars and performance data")
            st.write("Debugging information:")
            st.write(f"Unique hull IMOs: {len(hull_data['imo'].unique())}")
            st.write(f"Unique performance IMOs: {len(performance_data['vessel_imo'].unique())}")
            
            # Show overlap
            hull_imos = set(hull_data['imo'].unique())
            perf_imos = set(performance_data['vessel_imo'].unique())
            overlap = hull_imos.intersection(perf_imos)
            st.write(f"IMOs in both datasets: {len(overlap)}")
            if overlap:
                st.write(f"Sample overlapping IMOs: {list(overlap)[:5]}")
            st.stop()
        
        # Separate data into ballast and laden conditions
        ballast_df, laden_df = separate_data(combined_data)
        
        if ballast_df.empty:
            st.error("No ballast condition data found")
            st.stop()
            
        if laden_df.empty:
            st.error("No laden condition data found")
            st.stop()
        
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
        
        # Train models
        st.write("Training models...")
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
        st.write("Error details for debugging:")
        st.write(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())

# Add the "Run Test" button below the "Generate Predictions" button in the sidebar
if st.sidebar.button("Run Test"):
    with st.spinner("Running tests..."):
        test_results = run_all_tests()
    
    if test_results:
        st.subheader("Test Results Summary")
        for vessel_type, vessel_results in test_results.items():
            st.write(f"Vessel Type: {vessel_type}")
            for model_type, results_df in vessel_results.items():
                st.write(f"Model: {model_type}")
                st.dataframe(results_df.style.format("{:.2f}"))
                st.write("Mean values:")
                st.write(results_df.mean().to_frame().T.style.format("{:.2f}"))
                st.write("---")
    else:
        st.error("No test results were returned. Please check the logs for more information.")
