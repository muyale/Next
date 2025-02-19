# Customer Agent RL

## Overview
This project simulates an advanced customer behavior analysis using an RL agent (Deep Q-Network) based on a comprehensive synthetic dataset covering multiple sectors (Retail, E-commerce, Banking, Telecom, Travel). The dataset (30,000+ records) is generated using research insights and inserted into a local MongoDB database.

## File Structure
[AI_CUSTOMER_BEHAVIOR/
│── data/
│   └── customers.csv            # Generated synthetic customer records (30,000+)
│── models/
│   └── dqn_model.pth            # Saved RL model weights after training
│── src/
│   ├── generate_data.py         # Script to generate a comprehensive multi-sector dataset
│   ├── insert_data.py           # Script to load the CSV and insert data into MongoDB
│   ├── rl_agent.py              # Advanced Deep Q-Network (DQN) agent implementation (O³ model style)
│   └── dashboard.py             # Streamlit dashboard to visualize customer behavior insights
│── notebooks/
│   └── exploration.ipynb        # Notebook for exploratory analysis and experiments
│── requirements.txt             # Dependencies for the project
│── config.yaml                  # Configuration file for settings (MongoDB URI, hyperparameters, etc.)
└── README.md                    # Documentation and instructions


## Setup & Usage

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
