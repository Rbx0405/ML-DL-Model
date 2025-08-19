import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Example training data (you can expand this CSV-like dataset)
data = {
    "text": [
        "RTX 4060 graphics card", "RTX 4090 GPU", "Intel Iris integrated graphics",
        "Intel i3 processor", "Ryzen 3 entry level", "Intel i7 processor", "Ryzen 7 CPU",
        "Battery 6000mAh", "Battery 3000mAh", "Price 35000 INR", "Price 120000 INR"
    ],
    "label": [
        "High GPU", "High GPU", "Low GPU",
        "Low CPU", "Low CPU", "Good CPU", "Good CPU",
        "Long Battery", "Short Battery", "Budget", "Premium"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a pipeline (TF-IDF + Logistic Regression)
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Train the model
model.fit(df["text"], df["label"])

# Test
test_samples = [
    "Laptop with RTX 4060 and Intel i7",
    "Phone with 7000mAh battery",
    "Cheap phone under 20000",
    "Laptop with Intel i3 processor"
]

predictions = model.predict(test_samples)

for txt, pred in zip(test_samples, predictions):
    print(f"🔍 {txt}  --->  {pred}")