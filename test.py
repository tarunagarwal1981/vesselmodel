import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from database import get_db_engine
import sqlalchemy
from sqlalchemy.sql import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        return get_db_engine()
    except Exception as e:
        logger.error(f"Failed to get database connection: {str(e)}")
        raise

def get_hull_data(engine, vessel_type):
    query = text("""
    SELECT lpp, breadth, depth, deadweight, me_1_mcr_kw as mcr, load_type, imo
    FROM hull_particulars
    WHERE vessel_type = :vessel_type
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"vessel_type": vessel_type})
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except sqlalchemy.exc.ProgrammingError as e:
        logger.error(f"SQL error in get_hull_data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in get_hull_data: {str(e)}")
        raise

def get_performance_data(engine, imos):
    query = text("""
    SELECT speed_kts, me_consumption_mt, me_power_kw, load_type, vessel_imo
    FROM vessel_performance_model_data
    WHERE vessel_imo IN :imos
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"imos": tuple(imos)})
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except sqlalchemy.exc.ProgrammingError as e:
        logger.error(f"SQL error in get_performance_data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in get_performance_data: {str(e)}")
        raise

def separate_data(df):
    ballast_df = df[df['load_type'] == 'Ballast']
    laden_df = df[df['load_type'].isin(['Scantling', 'Design'])]
    laden_df = laden_df[laden_df['load_type'] == 'Scantling'] if 'Scantling' in laden_df['load_type'].values else laden_df
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
    elif model_type == "SVR":
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X_train_scaled, y_train)
    elif model_type == "Neural Network":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
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
    try:
        engine = get_db_connection()
        vessel_types = ["BULK CARRIER", "CONTAINER", "OIL TANKER"]
        model_types = [
            "Linear Regression with Polynomial Features",
            "Random Forest",
            "SVR",
            "Neural Network"
        ]
        all_results = {}

        for vessel_type in vessel_types:
            logger.info(f"Processing vessel type: {vessel_type}")
            hull_data = get_hull_data(engine, vessel_type)
            if hull_data.empty:
                logger.warning(f"No hull data found for vessel type: {vessel_type}")
                continue

            performance_data = get_performance_data(engine, hull_data['imo'].unique())
            if performance_data.empty:
                logger.warning(f"No performance data found for vessel type: {vessel_type}")
                continue

            combined_data = pd.merge(hull_data, performance_data, left_on='imo', right_on='vessel_imo')
            ballast_df, laden_df = separate_data(combined_data)

            input_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr', 'speed_kts']

            vessel_results = {}
            for model_type in model_types:
                logger.info(f"Training {model_type} model for {vessel_type}")
                models = {}
                for condition in ['ballast', 'laden']:
                    df = ballast_df if condition == 'ballast' else laden_df
                    for target in ['power', 'consumption']:
                        y = df['me_power_kw'] if target == 'power' else df['me_consumption_mt']
                        model, scaler = train_model(df[input_columns], y, model_type)
                        models[f"{condition}_{target}"] = (model, scaler)

                # Test on 10 random vessels
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

    except Exception as e:
        logger.error(f"Error in run_all_tests: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = run_all_tests()
        for vessel_type, models in results.items():
            print(f"\nResults for {vessel_type}")
            for model_type, df in models.items():
                print(f"\n{model_type}:")
                print(df)
                print(f"Mean % difference: {df.mean().mean():.2f}")
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
