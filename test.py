import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from database import get_db_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    return get_db_engine()

def get_hull_data(engine, vessel_type):
    query = """
    SELECT length_between_perpendiculars_m as lpp, breadth_moduled_m as breadth, 
           depth, deadweight, me_1_mcr_kw as mcr, imo, vessel_name, vessel_type
    FROM hull_particulars
    WHERE vessel_type = %(vessel_type)s
    """
    df = pd.read_sql(query, engine, params={'vessel_type': vessel_type})
    return remove_null_rows(df, "hull_data")

def get_performance_data(engine, imos):
    imos_string = ", ".join(f"'{str(imo)}'" for imo in imos)
    query = f"""
    SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo, load_type
    FROM vessel_performance_model_data
    WHERE vessel_imo::text IN ({imos_string})
    """
    df = pd.read_sql(query, engine)
    return remove_null_rows(df, "performance_data")

def remove_null_rows(df, df_name):
    initial_rows = len(df)
    df_cleaned = df.dropna()
    removed_rows = initial_rows - len(df_cleaned)
    logging.info(f"{df_name}: Removed {removed_rows} rows with null values. {len(df_cleaned)} rows remaining.")
    return df_cleaned

def separate_data(df):
    ballast_df = df[df['load_type'] == 'Ballast']
    laden_df = df[df['load_type'].isin(['Scantling', 'Design'])]
    laden_df = laden_df[laden_df['load_type'] == 'Scantling'] if 'Scantling' in laden_df['load_type'].values else laden_df
    return remove_null_rows(ballast_df, "ballast_data"), remove_null_rows(laden_df, "laden_data")

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
    
    return model, scaler

def predict(model, scaler, X, model_type):
    X_scaled = scaler.transform(X)
    if model_type == "Linear Regression with Polynomial Features":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)
        return model.predict(X_poly)
    else:
        return model.predict(X_scaled)

def calculate_percentage_difference(actual, predicted):
    return np.abs((actual - predicted) / actual) * 100

def run_all_tests():
    engine = get_db_connection()
    vessel_types = ["BULK CARRIER", "CONTAINER", "OIL TANKER"]
    model_types = ["Linear Regression with Polynomial Features", "Random Forest"]
    all_results = {}

    for vessel_type in vessel_types:
        logging.info(f"Processing {vessel_type}")
        hull_data = get_hull_data(engine, vessel_type)
        performance_data = get_performance_data(engine, hull_data['imo'].unique())
        
        combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo')
        combined_data = remove_null_rows(combined_data, "combined_data")
        
        ballast_df, laden_df = separate_data(combined_data)
        
        input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
        
        vessel_results = {}
        for model_type in model_types:
            logging.info(f"Training {model_type} model")
            models = {}
            for condition in ['ballast', 'laden']:
                df = ballast_df if condition == 'ballast' else laden_df
                for target in ['power', 'consumption']:
                    y = df['me_power_kw'] if target == 'power' else df['me_consumption_mt']
                    model, scaler = train_model(df[input_columns], y, model_type)
                    models[f"{condition}_{target}"] = (model, scaler)
            
            # Test on 10 random vessels or all vessels if less than 10
            test_vessels = hull_data.sample(n=min(10, len(hull_data)))
            speed_range = range(10, 23) if vessel_type == "CONTAINER" else range(8, 16)
            
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
                        actual = ballast_df if condition == 'ballast' else laden_df
                        actual = actual[(actual['imo'] == vessel['imo']) & 
                                        (actual['speed_kts'].round() == speed)]
                        
                        if not actual.empty:
                            for target in ['power', 'consumption']:
                                model, scaler = models[f"{condition}_{target}"]
                                predicted = predict(model, scaler, input_data, model_type)[0]
                                actual_value = actual['me_power_kw'].values[0] if target == 'power' else actual['me_consumption_mt'].values[0]
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
            
            df_results = pd.DataFrame(results).set_index('Speed (kts)')
            vessel_results[model_type] = df_results
        
        all_results[vessel_type] = vessel_results

    return all_results

if __name__ == "__main__":
    results = run_all_tests()
    for vessel_type, models in results.items():
        logging.info(f"\nResults for {vessel_type}")
        for model_type, df in models.items():
            logging.info(f"\n{model_type}:")
            logging.info(df)
