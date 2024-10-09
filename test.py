import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from database import get_db_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    try:
        return get_db_engine()
    except Exception as e:
        logging.error(f"Failed to establish database connection: {str(e)}")
        return None

# Model selection
MODEL_OPTIONS = ["Linear Regression with Polynomial Features", "Random Forest"]
VESSEL_TYPES = ["BULK CARRIER", "CONTAINER", "OIL TANKER"]

def get_hull_data(engine, vessel_type, limit=10):
    try:
        query = """
        SELECT length_between_perpendiculars_m as lpp, breadth_moduled_m as breadth, 
               depth, deadweight, me_1_mcr_kw as mcr, imo, vessel_name
        FROM hull_particulars
        WHERE vessel_type = %(vessel_type)s
        ORDER BY RANDOM() LIMIT %(limit)s
        """
        df = pd.read_sql(query, engine, params={'vessel_type': vessel_type, 'limit': limit})
        return df.dropna()
    except Exception as e:
        logging.error(f"Error fetching hull data for {vessel_type}: {str(e)}")
        return pd.DataFrame()

def get_performance_data(engine, imos):
    try:
        imos = [int(imo) for imo in imos if pd.notna(imo)]
        if not imos:
            logging.warning("No valid IMOs provided for performance data query.")
            return pd.DataFrame()
        
        query = """
        SELECT speed_kts, me_consumption_mt, me_power_kw, vessel_imo, load_type
        FROM vessel_performance_model_data
        WHERE vessel_imo IN %(imos)s
        """
        df = pd.read_sql(query, engine, params={'imos': tuple(imos)})
        return df.dropna()
    except Exception as e:
        logging.error(f"Error fetching performance data: {str(e)}")
        return pd.DataFrame()

def separate_data(df):
    try:
        ballast_df = df[df['load_type'] == 'Ballast']
        laden_df = df[df['load_type'].isin(['Scantling', 'Design'])]
        
        if 'Scantling' in laden_df['load_type'].values:
            laden_df = laden_df[laden_df['load_type'] == 'Scantling']
        elif 'Design' in laden_df['load_type'].values:
            laden_df = laden_df[laden_df['load_type'] == 'Design']
        else:
            laden_df = pd.DataFrame()  # Empty DataFrame if no matching load_type
        
        return ballast_df, laden_df
    except Exception as e:
        logging.error(f"Error separating data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def train_model(X, y, model_type):
    try:
        if X.empty or y.empty:
            logging.warning("Empty data provided for model training.")
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
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error training model {model_type}: {str(e)}")
        return None, None

def predict_performance(model, scaler, input_data, model_type):
    try:
        if model is None or scaler is None:
            return np.array([])
        
        input_scaled = scaler.transform(input_data)
        if model_type == "Linear Regression with Polynomial Features":
            poly = PolynomialFeatures(degree=2)
            input_poly = poly.fit_transform(input_scaled)
            prediction = model.predict(input_poly)
        else:
            prediction = model.predict(input_scaled)
        return np.maximum(0, prediction)
    except Exception as e:
        logging.error(f"Error predicting performance for {model_type}: {str(e)}")
        return np.array([])

def calculate_percentage_difference(actual, predicted):
    try:
        return np.abs((actual - predicted) / actual) * 100
    except ZeroDivisionError:
        logging.warning("Encountered zero division in percentage difference calculation.")
        return np.nan
    except Exception as e:
        logging.error(f"Error calculating percentage difference: {str(e)}")
        return np.nan

def run_test_for_vessel_type(engine, vessel_type, model_type):
    try:
        logging.info(f"Running test for {vessel_type} using {model_type}")
        
        test_vessels = get_hull_data(engine, vessel_type, limit=10)
        if test_vessels.empty:
            logging.warning(f"No test vessels found for {vessel_type}")
            return None
        
        performance_data = get_performance_data(engine, test_vessels['imo'].unique())
        if performance_data.empty:
            logging.warning(f"No performance data found for {vessel_type}")
            return None
        
        combined_data = pd.merge(test_vessels, performance_data, left_on='imo', right_on='vessel_imo')
        if combined_data.empty:
            logging.warning(f"No matching data found after merging for {vessel_type}")
            return None
        
        ballast_df, laden_df = separate_data(combined_data)
        
        if ballast_df.empty and laden_df.empty:
            logging.warning(f"Insufficient data after separation for {vessel_type}")
            return None
        
        input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
        
        models = {
            'ballast_power': train_model(ballast_df[input_columns], ballast_df['me_power_kw'], model_type) if not ballast_df.empty else (None, None),
            'ballast_consumption': train_model(ballast_df[input_columns], ballast_df['me_consumption_mt'], model_type) if not ballast_df.empty else (None, None),
            'laden_power': train_model(laden_df[input_columns], laden_df['me_power_kw'], model_type) if not laden_df.empty else (None, None),
            'laden_consumption': train_model(laden_df[input_columns], laden_df['me_consumption_mt'], model_type) if not laden_df.empty else (None, None)
        }
        
        results = {key: [] for key in models.keys()}
        
        for speed in range(8, 16):
            for _, vessel in test_vessels.iterrows():
                input_data = pd.DataFrame([[vessel['lpp'], vessel['breadth'], vessel['depth'], 
                                            vessel['deadweight'], vessel['mcr'], speed]], 
                                          columns=input_columns)
                
                for key, (model, scaler) in models.items():
                    if model is None or scaler is None:
                        continue
                    
                    prediction = predict_performance(model, scaler, input_data, model_type)
                    if prediction.size == 0:
                        continue
                    
                    df = ballast_df if 'ballast' in key else laden_df
                    actual = df[(df['imo'] == vessel['imo']) & (df['speed_kts'].round() == speed)]
                    
                    if not actual.empty:
                        actual_value = actual['me_power_kw'].values[0] if 'power' in key else actual['me_consumption_mt'].values[0]
                        diff = calculate_percentage_difference(actual_value, prediction[0])
                        results[key].append(diff)
        
        # Ensure all lists in results have the same length
        max_length = max(len(v) for v in results.values())
        results = {k: v + [np.nan] * (max_length - len(v)) for k, v in results.items()}
        
        results_df = pd.DataFrame({
            'Speed (kts)': range(8, 16),
            'Ballast Power % Diff': [np.nanmean(results['ballast_power'][i:i+10]) for i in range(0, len(results['ballast_power']), 10)],
            'Ballast Consumption % Diff': [np.nanmean(results['ballast_consumption'][i:i+10]) for i in range(0, len(results['ballast_consumption']), 10)],
            'Laden Power % Diff': [np.nanmean(results['laden_power'][i:i+10]) for i in range(0, len(results['laden_power']), 10)],
            'Laden Consumption % Diff': [np.nanmean(results['laden_consumption'][i:i+10]) for i in range(0, len(results['laden_consumption']), 10)]
        }).set_index('Speed (kts)')
        
        return results_df
    except Exception as e:
        logging.error(f"Error in run_test_for_vessel_type for {vessel_type} and {model_type}: {str(e)}")
        return None

def run_test():
    try:
        engine = get_db_connection()
        if engine is None:
            return None
        
        all_results = {}
        
        for vessel_type in VESSEL_TYPES:
            vessel_results = {}
            
            for model_type in MODEL_OPTIONS:
                results_df = run_test_for_vessel_type(engine, vessel_type, model_type)
                if results_df is not None and not results_df.empty:
                    vessel_results[model_type] = results_df
                else:
                    logging.warning(f"No valid results for {vessel_type} and {model_type}")
            
            if vessel_results:
                all_results[vessel_type] = vessel_results
        
        return all_results

    except Exception as e:
        logging.error(f"An error occurred during testing: {str(e)}")
        return None

if __name__ == "__main__":
    print("Running tests...")
    test_results = run_test()
    
    if test_results:
        for vessel_type, vessel_results in test_results.items():
            print(f"\nResults for {vessel_type}")
            
            for model_type, results_df in vessel_results.items():
                print(f"\nModel: {model_type}")
                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                    print(results_df)
                    print("\nMean values:")
                    print(results_df.mean())
                else:
                    print("No valid results for this model.")
    else:
        print("No test results were returned. Please check the logs for more information.")
