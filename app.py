import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("vehicle_price_prediction.pkl")  # your full pipeline with preprocessing + best model

# Page config
st.set_page_config(page_title="Vehicle Price Prediction", layout="wide")
st.title("Vehicle Price Prediction App")

st.write("""
This app predicts the **estimated market price** of a vehicle.
You can either enter the details manually or upload a CSV file.
""")

# Choose input method
input_method = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])

# ---------------- Manual Entry ----------------
if input_method == "Manual Entry":
    st.subheader("Enter Vehicle Details")

    with st.expander("Basic Info", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Vehicle Name", placeholder="e.g., Toyota Corolla")
            make = st.text_input("Make", placeholder="e.g., Toyota")
            model_name = st.text_input("Model", placeholder="e.g., Corolla")
            year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2020)
        with col2:
            mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, value=50000)
            fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid","Gasoline"])
            price = None  # Not used for prediction
            description = None

    with st.expander("Engine & Performance"):
        col1, col2 = st.columns(2)
        with col1:
            engine = st.number_input("Engine Size (in cc)", min_value=500, max_value=8000, value=1500)
            cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4)
        with col2:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"])

    with st.expander("Design & Features"):
        col1, col2 = st.columns(2)
        with col1:
            trim = st.text_input("Trim", placeholder="e.g., LE, SE")
            body = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Truck", "Van"])
            doors = st.number_input("Number of Doors", min_value=2, max_value=6, value=4)
        with col2:
            exterior_color = st.text_input("Exterior Color", placeholder="e.g., White")
            interior_color = st.text_input("Interior Color", placeholder="e.g., Black")

    # Create dataframe for prediction
    input_data = pd.DataFrame([[
        name, description, make, model_name, year, price, engine, cylinders, fuel, mileage,
        transmission, trim, body, doors, exterior_color, interior_color, drivetrain
    ]], columns=[
        'name', 'description', 'make', 'model', 'year', 'price', 'engine',
        'cylinders', 'fuel', 'mileage', 'transmission', 'trim', 'body', 'doors',
        'exterior_color', 'interior_color', 'drivetrain'
    ])

    if st.button("Predict Price"):
        try:
            prediction = pipeline.predict(input_data)[0]
            st.success(f"Estimated Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ---------------- CSV Upload ----------------
else:
    st.subheader("Upload CSV File")
    st.write("Make sure your CSV file contains all the required columns.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            predictions = pipeline.predict(data)
            data["Predicted Price"] = predictions
            st.success("Prediction completed!")
            st.dataframe(data)
            st.download_button("Download Predictions", data.to_csv(index=False), file_name="predicted_prices.csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")


