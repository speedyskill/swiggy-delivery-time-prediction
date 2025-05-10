import streamlit as st
import requests
from PIL import Image
import folium
from streamlit_folium import st_folium

# Swiggy logo (make sure swiggy_logo.png is in the same folder)
st.set_page_config(page_title="Swiggy Delivery Time Prediction", layout="centered")
logo = Image.open("Swiggy-logo.png")
st.image(logo, width=100)

st.title("Swiggy Delivery Time Prediction")

# Optional: Map for selecting delivery coordinates
with st.expander("📍 Select Delivery Location on Map (Optional)"):
    st.write("Click a point on the map to set Delivery Coordinates")

    # Map centered over India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    selected = st_folium(m, height=400, width=700)

    if selected.get("last_clicked"):
        delivery_lat = selected["last_clicked"]["lat"]
        delivery_lon = selected["last_clicked"]["lng"]
        st.success(f"Selected Location: Latitude={delivery_lat}, Longitude={delivery_lon}")
    else:
        delivery_lat, delivery_lon = None, None

st.markdown("### 🚚 Enter Delivery Details")

# Form for prediction input
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        ID = st.text_input("Order ID")
        Delivery_person_ID = st.text_input("Delivery Person ID")
        Delivery_person_Age = st.text_input("Delivery Person Age")
        Delivery_person_Ratings = st.text_input("Ratings")
        Restaurant_latitude = st.number_input("Restaurant Latitude", format="%.6f")
        Restaurant_longitude = st.number_input("Restaurant Longitude", format="%.6f")
        Delivery_location_latitude = st.number_input("Delivery Location Latitude", value=delivery_lat or 0.0, format="%.6f")
        Delivery_location_longitude = st.number_input("Delivery Location Longitude", value=delivery_lon or 0.0, format="%.6f")

    with col2:
        Order_Date = st.date_input("Order Date").strftime("%d-%m-%Y")
        Time_Orderd = st.text_input("Time Ordered", value="11:30 AM")
        Time_Order_picked = st.text_input("Time Picked", value="11:45 AM")
        Weatherconditions = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy", "NaN"])
        Road_traffic_density = st.selectbox("Traffic", ["Jam", "High", "Medium", "Low", "NaN"])
        Vehicle_condition = st.slider("Vehicle Condition", 0, 3, 1)
        Type_of_order = st.selectbox("Type of Order", ["Snack", "Meal", "Drinks", "Buffet"])
        Type_of_vehicle = st.selectbox("Vehicle Type", ["Motorcycle", "Scooter", "Bicycle", "Car"])
        multiple_deliveries = st.selectbox("Multiple Deliveries", ["1", "2", "3", "NaN"])
        Festival = st.selectbox("Festival", ["Yes", "No"])
        City = st.selectbox("City Type", ["Metropolitian", "Urban", "Semi-Urban"])

    submitted = st.form_submit_button("Predict")

# If user submits
if submitted:
    with st.spinner("Sending data for prediction..."):
        payload = {
            "ID": ID,
            "Delivery_person_ID": Delivery_person_ID,
            "Delivery_person_Age": Delivery_person_Age,
            "Delivery_person_Ratings": Delivery_person_Ratings,
            "Restaurant_latitude": Restaurant_latitude,
            "Restaurant_longitude": Restaurant_longitude,
            "Delivery_location_latitude": Delivery_location_latitude,
            "Delivery_location_longitude": Delivery_location_longitude,
            "Order_Date": Order_Date,
            "Time_Orderd": Time_Orderd,
            "Time_Order_picked": Time_Order_picked,
            "Weatherconditions": Weatherconditions,
            "Road_traffic_density": Road_traffic_density,
            "Vehicle_condition": Vehicle_condition,
            "Type_of_order": Type_of_order,
            "Type_of_vehicle": Type_of_vehicle,
            "multiple_deliveries": multiple_deliveries,
            "Festival": Festival,
            "City": City,
        }

        try:
            res = requests.post("http://localhost:8000/predict", json=payload)
            if res.status_code == 200:
                pred = res.json()  # This is a dictionary, so extract prediction and distance

                # Correct the access to prediction and distance
                estimated_time = round(pred['prediction'], 2)
                distance = round(pred['distance'], 2)

                st.success(f"🕐 Estimated Delivery Time: **{estimated_time} minutes**")
                st.success(f"📏 Delivery Distance: **{distance} km**")
            else:
                st.error(f"Error: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"Request Failed: {e}")