import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_purchase_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Enter the data below to get a prediction
""")
Age           = st.number_input("Age", 18, 100, 30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier      = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
mapping       = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
CityTier      = mapping[CityTier]
Occupation    = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender        = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, value=2)
PreferredPropertyStar  = st.number_input("Preferred Property Star", 1, 5, 3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, value=11)
Passport      = st.selectbox("Passport", ["No", "Yes"])
Passport      = 1 if Passport == "Yes" else 0
OwnCar        = st.selectbox("Own Car", ["No", "Yes"])
OwnCar        = 1 if OwnCar == "Yes" else 0
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, value=2)
Designation   = st.selectbox("Designation", ["Manager", "Senior Manager", "AVP", "VP", "Executive"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=25000)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 2)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Delux", "King", "Standard", "Super Deluxe"])
NumberOfFollowups = st.number_input("Number Of Followups", min_value=1, value=3)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
