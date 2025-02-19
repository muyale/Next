import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from dqn_agent import AdvancedDQNAgent

def load_data():
    """Load the dataset and return it as a DataFrame."""
    df = pd.read_csv("data/customers.csv")
    return df

def preprocess_data(df):
    """Normalize numerical features and handle missing values."""
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    # Scale numerical features
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, numerical_features

def customer_segmentation(df, numerical_features):
    """Perform K-Means clustering using available numerical features after handling NaN values."""
    if len(numerical_features) < 2:
        print("Not enough numerical features for segmentation. Skipping...")
        return df

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(df[numerical_features])

    return df

def churn_prediction(df):
    """Predict churn if the 'Churn' column exists."""
    if 'Churn' not in df.columns:
        print("Churn data not found. Skipping churn prediction.")
        return df
    
    # Ensure numerical features exist for prediction
    required_features = ['Age', 'Income', 'PurchaseFrequency', 'AvgSpend']
    available_features = [col for col in required_features if col in df.columns]

    if len(available_features) < 2:
        print("Insufficient churn data. Skipping churn prediction.")
        return df

    X = df[available_features].dropna()
    y = df['Churn'].dropna()

    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Not enough data after dropping NaNs. Skipping churn prediction.")
        return df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    df.loc[X.index, 'Churn_Predicted'] = model.predict(X)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Churn Prediction Model Accuracy: {accuracy:.2f}")

    return df

def personalized_marketing(df):
    """Generate sector-based marketing recommendations."""
    sector_recommendations = {
        "Retail": "Offer discounts on electronics & fashion.",
        "E-commerce": "Provide cart abandonment incentives.",
        "Banking": "Promote credit card rewards programs.",
        "Telecom": "Suggest better data plans based on usage.",
        "Travel": "Offer loyalty upgrades for frequent travelers."
    }

    if "Sector" in df.columns:
        df["Marketing_Recommendation"] = df["Sector"].map(sector_recommendations).fillna("General marketing approach")

    return df

def generate_agent_recommendations(agent, df, numerical_features):
    """Use the reinforcement learning agent to provide recommendations."""
    if not numerical_features:
        print("No numerical features available for AI recommendations. Skipping...")
        return df

    recommendations = []
    for _, row in df[numerical_features].iterrows():
        rec = agent.act(row.values)
        recommendations.append(rec)

    df["Agent_Recommendation"] = recommendations
    return df

def summarize_recommendations(df):
    """Summarize AI recommendations."""
    if "Agent_Recommendation" not in df.columns:
        print("No AI recommendations available to summarize.")
        return pd.DataFrame()

    summary = df["Agent_Recommendation"].value_counts().sort_index()
    summary_df = pd.DataFrame({
        "Action": ["Offer Discount (0)", "Recommend Product (1)", "No Action (2)"],
        "Count": [summary.get(0, 0), summary.get(1, 0), summary.get(2, 0)]
    })

    return summary_df

def NextAI():
    print(" Welcome to Next AI -  AI for Business Insights\n")
    
    df = load_data()
    print(f" Loaded dataset with {len(df)} records.")
    print(" Available columns:", df.columns.tolist(), "\n")

    df, numerical_features = preprocess_data(df)
    df = customer_segmentation(df, numerical_features)
    print(f" Segmentation complete. Sample:\n{df[['CustomerID', 'Segment']].head()}\n")

    df = churn_prediction(df)
    df = personalized_marketing(df)
    print(f"Marketing recommendations added. Sample:\n{df[['CustomerID', 'Marketing_Recommendation']].head()}\n")

    state_size = len(numerical_features)
    action_size = 3
    try:
        agent = AdvancedDQNAgent(state_size, action_size)
        df = generate_agent_recommendations(agent, df, numerical_features)
        print(" AI-generated recommendations added.\n")
    except Exception as e:
        print(f" AI Agent Initialization Failed: {e}")

    summary_df = summarize_recommendations(df)
    if not summary_df.empty:
        print(" AI Decision Summary:")
        print(summary_df.to_string(index=False), "\n")

    action_descriptions = {
        0: "Offer Discount",
        1: "Recommend Product",
        2: "No Action"
    }

    for action, description in action_descriptions.items():
        action_df = df[df["Agent_Recommendation"] == action][["CustomerID", "Agent_Recommendation"]].head(20)
        action_df["Agent_Recommendation"] = description
        print(f" First 20 Customers Receiving '{description}':")
        print(action_df.to_string(index=False), "\n")

    print(" Total Actions Taken:")
    print(summary_df.to_string(index=False), "\n")

    print(" Analysis complete! Thank you for using Next AI.")

if __name__ == "__main__":
    NextAI()


