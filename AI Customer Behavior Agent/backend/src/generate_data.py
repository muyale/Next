# File: src/generate_data.py
import pandas as pd
import random
import uuid
from faker import Faker

fake = Faker()

def generate_customer_record():
    customer_id = str(uuid.uuid4())
    age = random.randint(18, 80)
    gender = random.choice(["Male", "Female", "Other"])
    income = round(random.uniform(20000, 150000), 2)
    purchase_frequency = random.randint(1, 100)
    avg_spend = round(random.uniform(10, 2000), 2)
    churn_risk = random.choice(["Low", "Medium", "High"])
    
    # Sector and sector‑specific details
    sector = random.choice(["Retail", "E-commerce", "Banking", "Telecom", "Travel"])
    if sector in ["Retail", "E-commerce"]:
        product_category = random.choice(["Electronics", "Fashion", "Home", "Sports", "Beauty"])
        avg_rating = round(random.uniform(1, 5), 1)
        cart_abandon_rate = round(random.uniform(0, 0.5), 2)
        extra = {"Sector": sector, "ProductCategory": product_category,
                 "AvgRating": avg_rating, "CartAbandonRate": cart_abandon_rate}
    elif sector == "Banking":
        credit_score = random.randint(300, 850)
        num_transactions = random.randint(10, 200)
        extra = {"Sector": sector, "CreditScore": credit_score,
                 "NumTransactions": num_transactions}
    elif sector == "Telecom":
        monthly_data = round(random.uniform(0.5, 50), 2)
        call_minutes = random.randint(100, 3000)
        extra = {"Sector": sector, "MonthlyDataGB": monthly_data,
                 "CallMinutes": call_minutes}
    elif sector == "Travel":
        trips_per_year = random.randint(0, 15)
        loyalty_tier = random.choice(["Bronze", "Silver", "Gold", "Platinum"])
        extra = {"Sector": sector, "TripsPerYear": trips_per_year,
                 "LoyaltyTier": loyalty_tier}
    else:
        extra = {"Sector": sector}
    
    record = {
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "Income": income,
        "PurchaseFrequency": purchase_frequency,
        "AvgSpend": avg_spend,
        "ChurnRisk": churn_risk
    }
    record.update(extra)
    return record

def generate_dataset(num_records=30000):
    records = [generate_customer_record() for _ in range(num_records)]
    df = pd.DataFrame(records)
    # Save CSV to the data folder
    df.to_csv("data\\customers.csv", index=False)
    print(f" Generated {num_records} customer records and saved to ../data/customers.csv")

if __name__ == "__main__":
    generate_dataset()
