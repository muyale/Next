import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from backend.src.dqn_agent import AdvancedDQNAgent

st.set_page_config(page_title="Next (AI): Customer", layout="wide")
st.title("Next AI: Product Recommendation Agent")
st.markdown("Chat with nExT(AI) to get real-time customer targeting recommendations. Type a query (e.g., 'Show me discount recommendations') and nExT(AI) will reply with the corresponding customer table.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper function to load and preprocess data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data\\customers.csv")
    return df

def preprocess_data(df):
    churn_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df['ChurnRiskEncoded'] = df['ChurnRisk'].map(churn_mapping)
    features = df[['Age', 'Income', 'PurchaseFrequency', 'AvgSpend', 'ChurnRiskEncoded']].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

# Load and preprocess data
df = load_data()
states = preprocess_data(df)

# Setup RL agent (state vector of 5 features; actions: 0: Discount, 1: Recommend Product, 2: No Action)
state_size = states.shape[1]
action_size = 3
agent = AdvancedDQNAgent(state_size, action_size)

# For demonstration, run the agent on all customers to compute recommendations.
recommendations = [agent.act(state) for state in states]
df['Recommendation'] = recommendations

# Group recommendations
discount_df = df[df['Recommendation'] == 0]
product_df = df[df['Recommendation'] == 1]
no_action_df = df[df['Recommendation'] == 2]

# Define a simple function to process user queries
def process_query(query):
    query_lower = query.lower()
    if "discount" in query_lower:
        response = "Here are the customers recommended for a discount (Action 0):"
        table = discount_df[['CustomerID', 'Age', 'Income', 'PurchaseFrequency', 'AvgSpend', 'ChurnRisk']]
    elif "product" in query_lower:
        response = "Here are the customers recommended for a product suggestion (Action 1):"
        table = product_df[['CustomerID', 'Age', 'Income', 'PurchaseFrequency', 'AvgSpend', 'ChurnRisk']]
    elif "no action" in query_lower:
        response = "Here are the customers for whom no specific action is recommended (Action 2):"
        table = no_action_df[['CustomerID', 'Age', 'Income', 'PurchaseFrequency', 'AvgSpend', 'ChurnRisk']]
    elif "all" in query_lower or "recommendation" in query_lower:
        response = "Here are all customer recommendations:"
        table = df[['CustomerID', 'Age', 'Income', 'PurchaseFrequency', 'AvgSpend', 'ChurnRisk', 'Recommendation']]
    else:
        response = "I'm sorry, I didn't understand that. Please ask for discount, product, or no action recommendations."
        table = None
    return response, table

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**User:** {chat['message']}")
    else:
        st.markdown(f"**nExT(AI):** {chat['message']}")
        if chat.get("table") is not None:
            st.table(chat["table"])

# Input for new message
user_input = st.text_input("Type your message here and press Enter:")

if user_input:
    # Append user's message to chat history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    # Process the query and get reply
    reply_text, reply_table = process_query(user_input)
    st.session_state.chat_history.append({"role": "agent", "message": reply_text, "table": reply_table})
    # Clear input by simply relying on Streamlit's reactivity (the text input resets on script re-run)
    # Note: Without experiment
