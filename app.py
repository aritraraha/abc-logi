
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the delivery parameters to predict if there will be a delay.')

# Define input fields for the features
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=20.0, step=0.1)
traffic_congestion = st.slider('Traffic Congestion (1-5)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-3: 1=Good, 2=Moderate, 3=Bad)', min_value=1, max_value=3, value=2)
delivery_slot = st.slider('Delivery Slot (1-3: 1=Morning, 2=Afternoon, 3=Evening)', min_value=1, max_value=3, value=2)
driver_experience = st.number_input('Driver Experience (Years)', min_value=0, value=5, step=1)
num_stops = st.number_input('Number of Stops', min_value=0, value=3, step=1)
vehicle_age = st.number_input('Vehicle Age (Years)', min_value=0, value=2, step=1)
road_condition_score = st.slider('Road Condition Score (1-5: 1=Bad, 5=Good)', min_value=1, max_value=5, value=3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.1, value=5.0, step=0.1)
fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=0.1, value=10.0, step=0.1)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60, step=1)

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delay'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f'Likely to be DELAYED (Probability: {prediction_proba[1]:.2f})')
    else:
        st.success(f'Likely to be ON TIME (Probability: {prediction_proba[0]:.2f})')

    st.subheader('Prediction Probabilities:')
    st.write(f'On Time: {prediction_proba[0]:.2f}')
    st.write(f'Delayed: {prediction_proba[1]:.2f}')
