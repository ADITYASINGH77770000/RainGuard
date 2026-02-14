import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 1. LOAD DATA
# =========================

data = pd.read_csv("../data/processed/model_input.csv")

print("Initial data shape:", data.shape)
print(data.head())


# =========================
# 2. FIX CLASS IMBALANCE (IMPORTANT)
# =========================
# Hackathon-safe synthetic flood labeling
# Logic: High rainfall + saturated soil + low elevation → flood

data.loc[
    (data["Intensity"] > 80) &
    (data["soil_moisture_index"] > 0.6) &
    (data["elevation_risk"] > 0.6),
    "Flood Risk"
] = 1

print("\nFlood Risk distribution after adjustment:")
print(data["Flood Risk"].value_counts())


# =========================
# 3. SPLIT FEATURES & LABEL
# =========================

X = data[["Intensity", "soil_moisture_index", "elevation_risk"]]
y = data["Flood Risk"]


# =========================
# 4. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# 6. BUILD ANN MODEL
# =========================

model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# 7. TRAIN MODEL
# =========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# =========================
# 8. EVALUATE MODEL
# =========================

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# 9. SAVE MODEL (MODERN FORMAT)
# =========================

model.save("ann_flood_model.keras")
print("✅ ANN model saved as ann_flood_model.keras")


# =========================
# 10. QUICK DEMO PREDICTION
# =========================

sample = np.array([[120, 0.8, 0.65]])  # heavy rain scenario
sample_scaled = scaler.transform(sample)

risk_prob = model.predict(sample_scaled)[0][0]

print("\nDemo Prediction:")
print("Flood Risk Probability:", round(float(risk_prob), 3))
