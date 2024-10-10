import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from database import get_db_engine
import sqlalchemy

def get_db_connection():
    return get_db_engine()

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
    
    if model_type == "Linear":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_train_scaled)
        model = LinearRegression()
        model.fit(X_poly, y_train)
    elif model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict(model, scaler, X, model_type):
    X_scaled = scaler.transform(X)
    if model_type == "Linear":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)
        return model.predict(X_poly)
    else:
        return model.predict(X_scaled)

def calculate_percentage_difference(actual, predicted):
    return np.abs((actual - predicted) / actual) * 100

def run_tests():
    engine = get_db_connection()
    vessel_types = ["BULK CARRIER", "CONTAINER", "OIL TANKER"]
    results = {}

    for vessel_type in vessel_types:
        hull_data = get_hull_data(engine, vessel_type)
        performance_data = get_performance_data(engine, hull_data['imo'].unique())
        
        combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo')
        ballast_df, laden_df = separate_data(combined_data)
        
        input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']
        
        models = {}
        for condition in ['ballast', 'laden']:
            df = ballast_df if condition == 'ballast' else laden_df
            for target in ['power', 'consumption']:
                y = df['me_power_kw'] if target == 'power' else df['me_consumption_mt']
                for model_type in ['Linear', 'RandomForest']:
                    model, scaler = train_model(df[input_columns], y, model_type)
                    models[f"{condition}_{target}_{model_type}"] = (model, scaler)
        
        # Test on 10 random vessels
        test_vessels = hull_data.sample(n=10)
        speed_range = range(10, 23) if vessel_type == "CONTAINER" else range(8, 16)
        
        vessel_results = {model: {speed: {'power': [], 'consumption': []} for speed in speed_range} 
                          for model in ['Linear', 'RandomForest']}
        
        for _, vessel in test_vessels.iterrows():
            for speed in speed_range:
                input_data = pd.DataFrame([[vessel['lpp'], vessel['breadth'], vessel['depth'], 
                                            vessel['deadweight'], vessel['mcr'], speed]], 
                                          columns=input_columns)
                
                actual_ballast = ballast_df[(ballast_df['imo'] == vessel['imo']) & 
                                            (ballast_df['speed_kts'].round() == speed)]
                actual_laden = laden_df[(laden_df['imo'] == vessel['imo']) & 
                                        (laden_df['speed_kts'].round() == speed)]
                
                for model_type in ['Linear', 'RandomForest']:
                    for condition in ['ballast', 'laden']:
                        actual = actual_ballast if condition == 'ballast' else actual_laden
                        if not actual.empty:
                            for target in ['power', 'consumption']:
                                model, scaler = models[f"{condition}_{target}_{model_type}"]
                                predicted = predict(model, scaler, input_data, model_type)[0]
                                actual_value = actual['me_power_kw'].values[0] if target == 'power' else actual['me_consumption_mt'].values[0]
                                diff = calculate_percentage_difference(actual_value, predicted)
                                vessel_results[model_type][speed][target].append(diff)
        
        # Calculate average differences
        for model_type in ['Linear', 'RandomForest']:
            for speed in speed_range:
                for target in ['power', 'consumption']:
                    if vessel_results[model_type][speed][target]:
                        vessel_results[model_type][speed][target] = np.mean(vessel_results[model_type][speed][target])
                    else:
                        vessel_results[model_type][speed][target] = np.nan
        
        results[vessel_type] = vessel_results

    return results

def display_results(results):
    for vessel_type, vessel_results in results.items():
        print(f"\nResults for {vessel_type}")
        for model_type in ['Linear', 'RandomForest']:
            print(f"\n{model_type} Model:")
            df = pd.DataFrame(vessel_results[model_type]).T
            df.columns = ['Power % Diff', 'Consumption % Diff']
            df.index.name = 'Speed (kts)'
            print(df.to_string())

if __name__ == "__main__":
    results = run_tests()
    display_results(results)
