import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# Load dataset
df = pd.read_csv("Gold_Dataset.csv")

print("Dataset loaded")
print(df.head())
print(df.columns)


# Select features and target
X = df[["SPX", "USO", "SLV", "EUR/USD"]]
y = df["GLD"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model training
model = LinearRegression()

model.fit(X_train_scaled, y_train)


# Prediction and evaluation
y_pred = model.predict(X_test_scaled)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))


# Save model and scaler
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")


print("Model saved as best_model.pkl")
print("Scaler saved as scaler.pkl")