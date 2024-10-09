import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from database import get_db_engine

# Database setup
def get_db_connection():
    return get_db_engine()

# Sidebar for model selection
st.sidebar.header("Model Selection and Vessel Details")
model_options = ["Linear Regression with Polynomial Features", "Random Forest", "MLP Regressor"]
selected_model = st.sidebar.selectbox("Select a model to train:", model_options)

# User Inputs for Vessel Particulars
lpp = st.sidebar.number_input("Lpp (m)", min_value=50.0, max_value=400.0, step=0.1)
breadth = st.sidebar.number_input("Breadth (m)", min_value=10.0, max_value=100.0, step=0.1)
depth = st.sidebar.number_input("Depth (m)", min_value=5.0, max_value=50.0, step=0.1)
deadweight = st.sidebar.number_input("Deadweight (tons)", min_value=1000, max_value=500000, step=100)
year_of_built = st.sidebar.number_input("Year of Built", min_value=1900, max_value=2025, step=1)
vessel_type = st.sidebar.selectbox("Vessel Type", ["BULK CARRIER", "CONTAINER", "OIL TANKER"])
main_engine_make = st.sidebar.text_input("Main Engine Make")
main_engine_model = st.sidebar.text_input("Main Engine Model")
mcr = st.sidebar.number_input("MCR of Main Engine (kW)", min_value=500, max_value=100000, step=100)

# Function to get similar vessels from database
def get_similar_vessels(engine, lpp, breadth, depth, deadweight, vessel_type, mcr, threshold):
    query = """
    SELECT vessel_name, length_between_perpendiculars_m as lpp, breadth_moduled_m as breadth, 
           depth, deadweight, vessel_type, me_1_mcr_kw as mcr
    FROM hull_particulars
    WHERE
        length_between_perpendiculars_m BETWEEN %(lpp_min)s AND %(lpp_max)s AND
        breadth_moduled_m BETWEEN %(breadth_min)s AND %(breadth_max)s AND
        depth BETWEEN %(depth_min)s AND %(depth_max)s AND
        deadweight BETWEEN %(deadweight_min)s AND %(deadweight_max)s AND
        vessel_type = %(vessel_type)s AND
        me_1_mcr_kw BETWEEN %(mcr_min)s AND %(mcr_max)s
    """
    params = {
        'lpp_min': lpp * (1 - threshold/100),
        'lpp_max': lpp * (1 + threshold/100),
        'breadth_min': breadth * (1 - threshold/100),
        'breadth_max': breadth * (1 + threshold/100),
        'depth_min': depth * (1 - threshold/100),
        'depth_max': depth * (1 + threshold/100),
        'deadweight_min': deadweight * (1 - threshold/100),
        'deadweight_max': deadweight * (1 + threshold/100),
        'vessel_type': vessel_type,
        'mcr_min': mcr * (1 - threshold/100),
        'mcr_max': mcr * (1 + threshold/100)
    }
    return pd.read_sql(query, engine, params=params)

# Function to calculate confidence score
def calculate_confidence_score(similar_vessels, user_input):
    kpi_columns = ['lpp', 'breadth', 'depth', 'deadweight', 'mcr']
    vessel_deviations = []

    for _, vessel in similar_vessels.iterrows():
        kpi_deviations = []
        for col in kpi_columns:
            deviation = abs((vessel[col] - user_input[col]) / user_input[col]) * 100
            kpi_deviations.append(deviation)
        vessel_deviations.append(np.mean(kpi_deviations))

    average_deviation = np.mean(vessel_deviations)
    confidence_score = 100 - average_deviation
    return round(confidence_score, 2)

# Function to get speed, consumption, power data for selected vessels
def get_vessel_performance_data(engine, vessel_names):
    query = """
    SELECT VESSEL_NAME, speed_kts, me_power_kw, me_consumption_mt, displacement
    FROM vessel_performance_model_data
    WHERE vessel_name IN %(vessel_names_list)s
    """
    params = {
        'vessel_names_list': tuple([name.upper() for name in vessel_names])
    }
    return pd.read_sql(query, engine, params=params)

# Function to train the selected model
def train_model(X, y, model_type):
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

# Function to display color-coded confidence score
def display_confidence_score(score):
    if score >= 95:
        color = "green"
    elif 90 <= score < 95:
        color = "orange"
    else:
        color = "red"
    st.markdown(f"<p style='font-style: italic; font-weight: bold; color: {color};'>Confidence Score: {score}%</p>", unsafe_allow_html=True)

# Main execution
if st.sidebar.button("Fetch Data and Train Model"):
    engine = get_db_connection()
    
    # Automatic similarity threshold adjustment
    for threshold in [5, 10, 15, 20]:
        similar_vessels = get_similar_vessels(engine, lpp, breadth, depth, deadweight, vessel_type, mcr, threshold)
        if not similar_vessels.empty:
            st.write(f"Found {len(similar_vessels)} vessels matching the criteria with {threshold}% similarity threshold.")
            break
    
    if similar_vessels.empty:
        st.write("No vessels found matching the given criteria, even with a 20% similarity threshold.")
    else:
        # Calculate confidence score
        user_input = {'lpp': lpp, 'breadth': breadth, 'depth': depth, 'deadweight': deadweight, 'mcr': mcr}
        confidence_score = calculate_confidence_score(similar_vessels, user_input)
        display_confidence_score(confidence_score)
        
        st.write("Similar Vessels Details:")
        similar_vessels_display = similar_vessels.round({'lpp': 2, 'breadth': 2, 'depth': 2, 'deadweight': 2, 'mcr': 2})
        st.dataframe(similar_vessels_display.set_index('vessel_name'))
        
        vessel_names = similar_vessels['vessel_name'].tolist()
        
        df_performance = get_vessel_performance_data(engine, vessel_names)
        
        if df_performance.empty:
            st.write("No performance data available for the selected vessels.")
        else:
            X = df_performance[['speed_kts', 'displacement']]
            y_power = df_performance['me_power_kw']
            y_consumption = df_performance['me_consumption_mt']
            
            model_power = train_model(X, y_power, selected_model)
            model_consumption = train_model(X, y_consumption, selected_model)
            
            st.subheader("Output Tables (Predictions)")
            
            if vessel_type == "CONTAINER":
                output_speeds = range(10, 23)  # 10 to 22 knots
            else:
                output_speeds = range(8, 16)  # 8 to 15 knots
            
            ballast_displacement = df_performance['displacement'].min()
            laden_displacement = df_performance['displacement'].max()
            
            output_data_ballast = []
            output_data_laden = []
            
            for speed in output_speeds:
                for disp, data_list in [(ballast_displacement, output_data_ballast), (laden_displacement, output_data_laden)]:
                    if selected_model == "Linear Regression with Polynomial Features":
                        power = model_power.predict(PolynomialFeatures(degree=2).fit_transform([[speed, disp]]))[0]
                        consumption = model_consumption.predict(PolynomialFeatures(degree=2).fit_transform([[speed, disp]]))[0]
                    else:
                        power = model_power.predict([[speed, disp]])[0]
                        consumption = model_consumption.predict([[speed, disp]])[0]
                    data_list.append({
                        'Speed (kts)': speed,
                        'Power (kW)': round(power, 2),
                        'Consumption (mt/day)': round(consumption, 2)
                    })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Ballast Condition")
                st.dataframe(pd.DataFrame(output_data_ballast).set_index('Speed (kts)'))
            
            with col2:
                st.write("Laden Condition")
                st.dataframe(pd.DataFrame(output_data_laden).set_index('Speed (kts)'))

st.sidebar.write("Once the models are trained, you can analyze the predictions in the output tables.")
