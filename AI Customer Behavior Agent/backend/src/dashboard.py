import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# MongoDB settings (can also be loaded from config.yaml)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "customer_db"
COLLECTION_NAME = "customers"

def load_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    data = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(data)

st.title("📊 Advanced Multi-Sector Customer Behavior Dashboard")

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
age_min, age_max = st.sidebar.slider("Age Range", 18, 80, (25, 60))
income_min, income_max = st.sidebar.slider("Income Range", 20000, 150000, (30000, 100000))
sectors = st.sidebar.multiselect("Select Sector(s)", options=df["Sector"].unique(), default=df["Sector"].unique())

filtered_df = df[
    (df["Age"] >= age_min) & (df["Age"] <= age_max) &
    (df["Income"] >= income_min) & (df["Income"] <= income_max) &
    (df["Sector"].isin(sectors))
]

st.write(f"### Showing {len(filtered_df)} records based on filters")
st.dataframe(filtered_df.head(50))

# Plot: Distribution by Sector
st.subheader("Customer Distribution by Sector")
sector_counts = filtered_df["Sector"].value_counts()
fig, ax = plt.subplots()
sector_counts.plot(kind="bar", ax=ax)
st.pyplot(fig)

# Sector-specific insights
st.subheader("Sector-specific Insights")
for sector in sectors:
    st.write(f"**{sector}**")
    sector_data = filtered_df[filtered_df["Sector"] == sector]
    if sector in ["Retail", "E-commerce"]:
        avg_rating = sector_data["AvgRating"].mean() if "AvgRating" in sector_data.columns else None
        st.write(f"Average Product Rating: {avg_rating:.2f}" if avg_rating else "No rating data")
    elif sector == "Banking":
        avg_credit = sector_data["CreditScore"].mean() if "CreditScore" in sector_data.columns else None
        st.write(f"Average Credit Score: {avg_credit:.2f}" if avg_credit else "No credit data")
    elif sector == "Telecom":
        avg_data = sector_data["MonthlyDataGB"].mean() if "MonthlyDataGB" in sector_data.columns else None
        st.write(f"Average Monthly Data Usage: {avg_data:.2f} GB" if avg_data else "No telecom data")
    elif sector == "Travel":
        avg_trips = sector_data["TripsPerYear"].mean() if "TripsPerYear" in sector_data.columns else None
        st.write(f"Average Trips per Year: {avg_trips:.2f}" if avg_trips else "No travel data")
    st.write("---")

st.write("Dashboard powered by **Streamlit, MongoDB & an advanced multi-sector synthetic dataset**")
