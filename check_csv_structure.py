import os
import pandas as pd

# Get absolute path of the folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full file paths
laptop_path = os.path.join(BASE_DIR, "laptops.csv")
phone_path = os.path.join(BASE_DIR, "Phone.csv")

# Load datasets
laptop_df = pd.read_csv(laptop_path)
phone_df = pd.read_csv(phone_path)

print("=== LAPTOP CSV STRUCTURE ===")
print("Shape:", laptop_df.shape)
print("Columns:", laptop_df.columns.tolist())
print("\nFirst few rows:")
print(laptop_df.head())

print("\n=== PHONE CSV STRUCTURE ===")
print("Shape:", phone_df.shape)
print("Columns:", phone_df.columns.tolist())
print("\nFirst few rows:")
print(phone_df.head())