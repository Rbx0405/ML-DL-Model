import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Get absolute path of the folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full file paths
laptop_path = os.path.join(BASE_DIR, "laptops.csv")
phone_path = os.path.join(BASE_DIR, "Phone.csv")

# Load datasets
laptop_df = pd.read_csv(laptop_path)
phone_df = pd.read_csv(phone_path)

print("Laptop dataset loaded with shape:", laptop_df.shape)
print("Phone dataset loaded with shape:", phone_df.shape)

# Function to train a model on a given dataset
def train_model(df):
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(df["text"], df["label"])
    return model

# Train models separately
laptop_model = train_model(laptop_df)
phone_model = train_model(phone_df)

# Function to detect device type (rule-based)
def detect_device(query: str):
    query_lower = query.lower()
    if "laptop" in query_lower or "notebook" in query_lower or "pc" in query_lower:
        return "laptop"
    elif "phone" in query_lower or "mobile" in query_lower or "smartphone" in query_lower:
        return "phone"
    else:
        # Default choice if unsure
        return "phone"

# Function to predict based on query
def predict_query(query: str):
    device = detect_device(query)
    if device == "laptop":
        prediction = laptop_model.predict([query])[0]
    else:
        prediction = phone_model.predict([query])[0]
    return device, prediction

# Example test queries
test_samples = [
    "Laptop with RTX 4060 and Intel i7",
    "Phone with 7000mAh battery",
    "Cheap phone under 20000",
    "Laptop with Intel i3 processor"
]

for txt in test_samples:
    device, pred = predict_query(txt)
    print(f"{txt} ---> ({device.upper()}) {pred}")