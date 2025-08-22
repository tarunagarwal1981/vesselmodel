import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from database import get_db_engine
import sqlalchemy

def get_db_connection():
    engine = get_db_engine()
    
    # Configure engine with timeouts
    if hasattr(engine, 'pool'):
        engine.pool._timeout = 300  # 5 minutes
        engine.pool._recycle = 3600  # 1 hour
    
    return engine

def get_hull_data(engine, vessel_type):
    """Get hull data with the same logic as the main app"""
    try:
        # Simple count query first
        count_query = sqlalchemy.text("SELECT COUNT(*) FROM hull_particulars WHERE vessel_type = :vessel_type")
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SET statement_timeout = '60s'"))
            result = conn.execute(count_query, {'vessel_type': vessel_type})
            count = result.scalar()
            print(f"Found {count} total records for {vessel_type}")
        
        if count == 0:
            return pd.DataFrame()
        
        # Main query
        query = sqlalchemy.text("""
        SELECT 
            "Length_between_perpendiculars_m" as lpp, 
            "Breadth_Moduled_m" as breadth, 
            "Depth" as depth, 
            deadweight, 
            "ME_1_MCR_kW" as mcr, 
            "IMO" as imo, 
            "Vessel_Name" as vessel_name
        FROM hull_particulars
        WHERE vessel_type = :vessel_type
        LIMIT 1000
        """)
        
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SET statement_timeout = '120s'"))
            result = conn.execute(query, {'vessel_type': vessel_type})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            return df
        
        # Convert string columns to numeric where needed
        numeric_columns = ['lpp', 'breadth', 'depth', 'mcr', 'deadweight']
        for col in numeric_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure IMO is string for consistent merging
        if 'imo' in df.columns:
            df['imo'] = df['imo'].astype(str).str.strip()
        
        # Filter out rows with null values in critical columns
        df_clean = df.dropna(subset=['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'imo'])
        print(f"After cleaning: {len(df_clean)} usable records")
        
        return df_clean
        
    except Exception as e:
        print(f"Error fetching hull data: {str(e)}")
        return pd.DataFrame()

def get_performance_data(engine, imos):
    """Get performance data with the same logic as the main app"""
    if len(imos) == 0:
        return pd.DataFrame()
        
    imos = [str(imo) for imo in imos]
    
    try:
        # Limit the number of IMOs to prevent timeout
        limited_imos = imos[:50]
        
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
        print(f"Error fetching performance data: {str(e)}")
        return pd.DataFrame()

def separate_data(df):
    """Separate data into ballast and laden conditions"""
    ballast_df = df[df['load_type'] == 'Ballast']
    laden_df = df[df['load_type'].isin(['Scantling', 'Design'])]
    
    if 'Scantling' in laden_df['load_type'].values:
        laden_df = laden_df[laden_df['load_type'] == 'Scantling']
    else:
        laden_df = laden_df[laden_df['load_type'] == 'Design']
    
    return ballast_df, laden_df

def train_model(X, y, model_type):
    """Train model with given data and model type"""
    if len(X) < 4:  # Need at least 4 samples for train_test_split
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
    elif model_type == "MLP Regressor":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_performance(model, scaler, input_data, model_type):
    """Make predictions with the trained model"""
    input_scaled = scaler.transform(input_data)
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        input_poly = poly.fit_transform(input_scaled)
        prediction = model.predict(input_poly)
    else:
        prediction = model.predict(input_scaled)
    return np.maximum(0, prediction)

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def test_model_performance(vessel_type, model_type):
    """Test model performance for a specific vessel type and model"""
    try:
        engine = get_db_connection()
        
        # Get data
        hull_data = get_hull_data(engine, vessel_type)
        if hull_data.empty:
            print(f"No hull data found for {vessel_type}")
            return None
            
        performance_data = get_performance_data(engine, hull_data['imo'].unique())
        if performance_data.empty:
            print(f"No performance data found for {vessel_type}")
            return None
        
        # Fix data type mismatch before merging
        hull_data['imo'] = hull_data['imo'].astype(str)
        performance_data['vessel_imo'] = performance_data['vessel_imo'].astype(str)
        
        # Merge data
        combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo').dropna()
        if combined_data.empty:
            print(f"No matching data found for {vessel_type}")
            return None
        
        # Separate data
        ballast_df, laden_df = separate_data(combined_data)
        
        if ballast_df.empty or laden_df.empty:
            print(f"Insufficient data for {vessel_type} - ballast: {len(ballast_df)}, laden: {len(laden_df)}")
            return None
        
        input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
        
        results = {}
        
        # Test ballast condition models
        if len(ballast_df) >= 4:
            # Power prediction
            ballast_power_model, ballast_power_scaler = train_model(
                ballast_df[input_columns], ballast_df['me_power_kw'], model_type
            )
            if ballast_power_model is not None:
                ballast_power_pred = predict_performance(
                    ballast_power_model, ballast_power_scaler, 
                    ballast_df[input_columns], model_type
                )
                results['Ballast_Power'] = calculate_metrics(ballast_df['me_power_kw'], ballast_power_pred)
            
            # Consumption prediction
            ballast_consumption_model, ballast_consumption_scaler = train_model(
                ballast_df[input_columns], ballast_df['me_consumption_mt'], model_type
            )
            if ballast_consumption_model is not None:
                ballast_consumption_pred = predict_performance(
                    ballast_consumption_model, ballast_consumption_scaler,
                    ballast_df[input_columns], model_type
                )
                results['Ballast_Consumption'] = calculate_metrics(ballast_df['me_consumption_mt'], ballast_consumption_pred)
        
        # Test laden condition models
        if len(laden_df) >= 4:
            # Power prediction
            laden_power_model, laden_power_scaler = train_model(
                laden_df[input_columns], laden_df['me_power_kw'], model_type
            )
            if laden_power_model is not None:
                laden_power_pred = predict_performance(
                    laden_power_model, laden_power_scaler,
                    laden_df[input_columns], model_type
                )
                results['Laden_Power'] = calculate_metrics(laden_df['me_power_kw'], laden_power_pred)
            
            # Consumption prediction
            laden_consumption_model, laden_consumption_scaler = train_model(
                laden_df[input_columns], laden_df['me_consumption_mt'], model_type
            )
            if laden_consumption_model is not None:
                laden_consumption_pred = predict_performance(
                    laden_consumption_model, laden_consumption_scaler,
                    laden_df[input_columns], model_type
                )
                results['Laden_Consumption'] = calculate_metrics(laden_df['me_consumption_mt'], laden_consumption_pred)
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        print(f"Error testing {vessel_type} with {model_type}: {str(e)}")
        return None

def run_all_tests():
    """Run tests for all vessel types and models"""
    vessel_types = ["BULK CARRIER", "CONTAINER", "OIL TANKER"]
    model_types = ["Linear Regression with Polynomial Features", "Random Forest", "MLP Regressor"]
    
    all_results = {}
    
    for vessel_type in vessel_types:
        print(f"\nTesting {vessel_type}...")
        vessel_results = {}
        
        for model_type in model_types:
            print(f"  Testing {model_type}...")
            results = test_model_performance(vessel_type, model_type)
            if results is not None:
                vessel_results[model_type] = results
            else:
                print(f"    No results for {model_type}")
        
        if vessel_results:
            all_results[vessel_type] = vessel_results
        else:
            print(f"  No results for {vessel_type}")
    
    return all_results

if __name__ == "__main__":
    results = run_all_tests()
    
    if results:
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        for vessel_type, vessel_results in results.items():
            print(f"\nVessel Type: {vessel_type}")
            for model_type, results_df in vessel_results.items():
                print(f"\nModel: {model_type}")
                print(results_df.round(4))
                print(f"Mean values:")
                print(results_df.mean().round(4))
    else:
        print("No test results generated.")
