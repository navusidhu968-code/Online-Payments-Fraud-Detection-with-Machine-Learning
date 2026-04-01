import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Dataset load karo (apni file ka naam change karo)
data = pd.read_csv("data.csv")

# Features aur target define karo
X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model train karo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model save karo
joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")
