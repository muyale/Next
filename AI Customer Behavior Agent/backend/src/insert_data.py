# File: src/insert_data.py
import pandas as pd
from pymongo import MongoClient

# Configuration (can also be set via config.yaml)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "customer_db"
COLLECTION_NAME = "customers"

def insert_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Load data from CSV
    df = pd.read_csv("data\\customers.csv")
    records = df.to_dict(orient="records")
    collection.delete_many({})  
    collection.insert_many(records)
    print(f" Inserted {len(records)} records into MongoDB collection '{COLLECTION_NAME}' in DB '{DB_NAME}'.")

if __name__ == "__main__":
    insert_data()
