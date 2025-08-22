import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from database import get_db_engine
import sqlalchemy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    engine = get_db_engine()
    
    # Configure engine with timeouts
    if hasattr(engine, 'pool'):
        engine.pool._timeout = 300  # 5 minutes
        engine.pool._recycle = 3600  # 1 hour
    
    return engine

def get_hull_data(engine, vessel_type):
    try:
        query = sqlalchemy.text("""
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
        WHERE vessel_type = :vessel_type
        LIMIT 1000
        """)
        
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SET statement_timeout = '120s'"))
            result = conn.execute(query, {'vessel_type': vessel_type})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            logging.warning(f"No data found for vessel type: {vessel_type}")
            return df
        
        # Convert string columns to numeric where needed
        numeric_columns = ['lpp', 'breadth', 'depth', 'mcr', 'deadweight']
        for col in numeric_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure IMO is string for consistent merging
        if 'imo' in df.columns:
            df['imo'] = df['imo'].astype(str).str.strip()
        
        return remove_null_rows(df, "hull_data")
        
    except Exception as e:
        logging.error(f"Error fetching hull data: {str(e)}")
        return pd.DataFrame()

def get_performance_data(engine, imos):
    if len(imos) == 0:
        return pd.DataFrame()
        
    try:
        imos = [str(imo) for imo in imos]  # Convert to string
        limited_imos = imos[:50]  # Limit to prevent timeout
        
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
        
        if df.empty:
            logging.warning("No performance data found")
            return df
        
        # Ensure vessel_imo is string for consistent merging
        if 'vessel_imo' in df.columns:
            df['vessel_imo'] = df['vessel_imo'].astype(str).str.strip()
        
        return remove_null_rows(df, "performance_data")
        
    except Exception as e:
        logging.error(f"Error fetching performance data: {str(e)}")
        return pd.DataFrame()

def remove_null_rows(df, df_name):
    if df.empty:
        return df
        
    initial_rows = len(df)
    df_cleaned = df.dropna()
    removed_rows = initial_rows - len(df_cleaned)
    logging.info(f"{df_name}: Removed {removed_rows} rows with null values. {len(df_cleaned)} rows remaining.")
    return df_cleaned

def separate_data(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    ballast_df = df[df['load_type'] == 'Ballast']
    laden_df = df[df['load_type'].isin(['Scantling', 'Design'])]
    
    if 'Scantling' in laden_df['load_type'].values:
        laden_df = laden_df[laden_df['load_type'] == 'Scantling']
    else:
        laden_df = laden_df[laden_df['load_type'] == 'Design']
    
    return remove_null_rows(ballast_df, "ballast_data"), remove_null_rows(laden_df, "laden_data")

def train_model(X, y, model_type):
    if len(X) < 5:  # Need minimum data for training
        logging.warning(f"Insufficient data for training: {len(X)} samples")
        return None, None
        
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
    
    return model, scaler

def predict(model, scaler, X, model_type):
    if model is None or scaler is None:
        return np.array([0])
        
    X_scaled = scaler.transform(X)
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)
        return model.predict(X_poly)
    else:
        return model.predict(X_scaled)

def calculate_percentage_difference(actual, predicted):
    if actual == 0:
        return 0 if predicted == 0 else 100
    return np.abs((actual - predicted) / actual) * 100

def run_all_tests():
    try:
        engine = get_db_connection()
        vessel_types = ["BULK CARRIER", "CONTAINER", "OIL TANKER"]
        model_types = ["Linear Regression with Polynomial Features", "Random Forest"]
        all_results = {}

        for vessel_type in vessel_types:
            logging.info(f"Processing {vessel_type}")
            
            # Get hull data
            hull_data = get_hull_data(engine, vessel_type)
            if hull_data.empty:
                logging.warning(f"No hull data found for {vessel_type}")
                continue
            
            # Get performance data
            performance_data = get_performance_data(engine, hull_data['imo'].unique())
            if performance_data.empty:
                logging.warning(f"No performance data found for {vessel_type}")
                continue
            
            # Merge data
            combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo')
            combined_data = remove_null_rows(combined_data, "combined_data")
            
            if combined_data.empty:
                logging.warning(f"No combined data for {vessel_type}")
                continue
            
            # Separate ballast and laden data
            ballast_df, laden_df = separate_data(combined_data)
            
            if ballast_df.empty and laden_df.empty:
                logging.warning(f"No separated data for {vessel_type}")
                continue
            
            input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
            
            vessel_results = {}
            for model_type in model_types:
                logging.info(f"Training {model_type} model for {vessel_type}")
                
                # Train models
                models = {}
                for condition in ['ballast', 'laden']:
                    df = ballast_df if condition == 'ballast' else laden_df
                    if df.empty:
                        logging.warning(f"No {condition} data for {vessel_type}")
                        continue
                        
                    for target in ['power', 'consumption']:
                        target_col = 'me_power_kw' if target == 'power' else 'me_consumption_mt'
                        if target_col not in df.columns:
                            logging.warning(f"Column {target_col} not found")
                            continue
                            
                        y = df[target_col]
                        model, scaler = train_model(df[input_columns], y, model_type)
                        models[f"{condition}_{target}"] = (model, scaler)
                
                if not models:
                    logging.warning(f"No models trained for {vessel_type} with {model_type}")
                    continue
                
                # Test predictions
                test_vessels = hull_data.sample(n=min(5, len(hull_data)))  # Reduced to 5 for faster testing
                speed_range = range(10, 16) if vessel_type == "CONTAINER" else range(8, 14)  # Reduced range
                
                results = []
                for speed in speed_range:
                    ballast_power_diffs = []
                    ballast_consumption_diffs = []
                    laden_power_diffs = []
                    laden_consumption_diffs = []
                    
                    for _, vessel in test_vessels.iterrows():
                        input_data = pd.DataFrame([[vessel['lpp'], vessel['breadth'], vessel['depth'], 
                                                    vessel['deadweight'], vessel['mcr'], speed]], 
                                                  columns=input_columns)
                        
                        for condition in ['ballast', 'laden']:
                            actual_df = ballast_df if condition == 'ballast' else laden_df
                            if actual_df.empty:
                                continue
                                
                            actual = actual_df[(actual_df['imo'] == vessel['imo']) & 
                                            (abs(actual_df['speed_kts'] - speed) <= 1)]  # Allow ±1 knot tolerance
                            
                            if not actual.empty:
                                for target in ['power', 'consumption']:
                                    model_key = f"{condition}_{target}"
                                    if model_key not in models:
                                        continue
                                        
                                    model, scaler = models[model_key]
                                    if model is None:
                                        continue
                                        
                                    predicted = predict(model, scaler, input_data, model_type)[0]
                                    target_col = 'me_power_kw' if target == 'power' else 'me_consumption_mt'
                                    actual_value = actual[target_col].iloc[0]
                                    diff = calculate_percentage_difference(actual_value, predicted)
                                    
                                    if condition == 'ballast':
                                        if target == 'power':
                                            ballast_power_diffs.append(diff)
                                        else:
                                            ballast_consumption_diffs.append(diff)
                                    else:
                                        if target == 'power':
                                            laden_power_diffs.append(diff)
                                        else:
                                            laden_consumption_diffs.append(diff)
                    
                    results.append({
                        'Speed (kts)': speed,
                        'Ballast Power % Diff from Actual': np.mean(ballast_power_diffs) if ballast_power_diffs else np.nan,
                        'Ballast Consumption % Diff from Actual': np.mean(ballast_consumption_diffs) if ballast_consumption_diffs else np.nan,
                        'Laden Power % Diff from Actual': np.mean(laden_power_diffs) if laden_power_diffs else np.nan,
                        'Laden Consumption % Diff from Actual': np.mean(laden_consumption_diffs) if laden_consumption_diffs else np.nan
                    })
                
                if results:
                    df_results = pd.DataFrame(results).set_index('Speed (kts)')
                    vessel_results[model_type] = df_results
                else:
                    logging.warning(f"No test results for {vessel_type} with {model_type}")
            
            if vessel_results:
                all_results[vessel_type] = vessel_results

        return all_results
        
    except Exception as e:
        logging.error(f"Error in run_all_tests: {str(e)}")
        return {}

if __name__ == "__main__":
    results = run_all_tests()
    for vessel_type, models in results.items():
        logging.info(f"\nResults for {vessel_type}")
        for model_type, df in models.items():
            logging.info(f"\n{model_type}:")
            logging.info(df)
