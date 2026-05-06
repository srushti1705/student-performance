"""
AI-Based Student Academic Monitoring System
Enhanced Training Pipeline with Risk Scoring and Interventions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ACADEMIC RISK PREDICTION MODEL TRAINING")
print("=" * 80)

# 1. LOAD AND EXPLORE DATASET
print("\n[1] Loading student lifestyle dataset...")
df = pd.read_csv('student_lifestyle_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")

# 2. DATA PREPROCESSING & FEATURE ENGINEERING
print("\n[2] Preprocessing data and engineering features...")

# Remove Student_ID
df = df.drop('Student_ID', axis=1)

# Create feature engineering for institutional metrics
df['Study_Consistency'] = df['Study_Hours_Per_Day'].apply(
    lambda x: min(100, (x / 8) * 100)  # Normalize to 0-100 based on 8 hours ideal
)
df['Sleep_Quality'] = df['Sleep_Hours_Per_Day'].apply(
    lambda x: min(100, (x / 8) * 100)  # 7-8 hours is ideal
)
df['Lifestyle_Balance'] = (
    df['Study_Consistency'] * 0.3 +
    df['Sleep_Quality'] * 0.3 +
    (100 - df['Social_Hours_Per_Day'].apply(lambda x: min(100, x * 12.5))) * 0.2 +
    df['Physical_Activity_Hours_Per_Day'].apply(lambda x: min(100, x * 16.7)) * 0.2
)
df['Screen_Time_Risk'] = df['Social_Hours_Per_Day'].apply(
    lambda x: min(100, x * 10)  # Higher = more risky
)
df['Stress_Score'] = df['Stress_Level'].map({
    'Low': 20, 'Moderate': 50, 'High': 90
})

# Create performance categories (Risk Levels)
print("  Creating risk categories...")
q3 = df['GPA'].quantile(0.75)
q1 = df['GPA'].quantile(0.25)

def categorize_risk(gpa):
    """
    0 = Low Risk (High GPA)
    1 = Moderate Risk (Average GPA)
    2 = High Risk (Low GPA - at-risk)
    """
    if gpa >= q3:
        return 0  # Low Risk
    elif gpa >= q1:
        return 1  # Moderate Risk
    else:
        return 2  # High Risk

df['Risk_Category'] = df['GPA'].apply(categorize_risk)

# Feature importance derived metrics
df['Productivity_Index'] = (
    (df['Study_Hours_Per_Day'] / df['Study_Hours_Per_Day'].max()) * 0.4 +
    (df['Sleep_Hours_Per_Day'] / 8) * 0.3 +
    ((8 - df['Social_Hours_Per_Day']) / 8) * 0.3
) * 100

df['Academic_Consistency'] = df['GPA'].rolling(window=1).mean() * 20  # Normalized

# Select features for model
feature_cols = [
    'Study_Hours_Per_Day',
    'Sleep_Hours_Per_Day',
    'Social_Hours_Per_Day',
    'Extracurricular_Hours_Per_Day',
    'Physical_Activity_Hours_Per_Day',
    'Stress_Score',
    'Study_Consistency',
    'Sleep_Quality',
    'Lifestyle_Balance',
    'Screen_Time_Risk',
    'Productivity_Index'
]

X = df[feature_cols].copy()
y = df['Risk_Category'].copy()

print(f"  Features selected: {len(feature_cols)}")
print(f"  Risk distribution:")
print(f"    Low Risk (0): {(y == 0).sum()} students")
print(f"    Moderate Risk (1): {(y == 1).sum()} students")
print(f"    High Risk (2): {(y == 2).sum()} students")

# 3. TRAIN-TEST SPLIT
print("\n[3] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# 4. FEATURE SCALING
print("\n[4] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  Scaler saved as scaler.pkl")

# 5. BUILD ENHANCED ANN MODEL
print("\n[5] Building enhanced neural network model...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dropout(0.1),
    
    Dense(3, activation='softmax')  # 3 classes: Low, Moderate, High Risk
])

print("  Model architecture:")
print("    Input Layer: 11 features")
print("    Hidden Layer 1: 64 neurons + Batch Norm + Dropout(0.3)")
print("    Hidden Layer 2: 32 neurons + Batch Norm + Dropout(0.2)")
print("    Hidden Layer 3: 16 neurons + Dropout(0.1)")
print("    Output Layer: 3 neurons (softmax)")

# 6. COMPILE MODEL
print("\n[6] Compiling model...")
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
print("\n[7] Computing class weights for imbalanced data...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"  Class weights: {class_weight_dict}")

# 8. TRAIN MODEL
print("\n[8] Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# 9. EVALUATE MODEL
print("\n[9] Evaluating model on test set...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy * 100:.2f}%")

# Get predictions
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, 
    target_names=['Low Risk', 'Moderate Risk', 'High Risk']))

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 10. SAVE MODEL
print("\n[10] Saving trained model...")
model.save('model.h5')
print("  Model saved as model.h5")

# 11. SAVE FEATURE METADATA
print("\n[11] Saving feature metadata...")
feature_metadata = {
    'features': feature_cols,
    'risk_categories': {
        '0': 'Low Risk',
        '1': 'Moderate Risk',
        '2': 'High Risk'
    },
    'model_accuracy': float(accuracy),
    'model_loss': float(loss),
    'feature_ranges': {
        feature: {
            'min': float(df[feature].min()),
            'max': float(df[feature].max()),
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std())
        }
        for feature in feature_cols
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)
print("  Metadata saved as model_metadata.json")

# 12. SAVE THRESHOLD DATA FOR RISK SCORING
print("\n[12] Computing risk score thresholds...")
df['Risk_Score'] = 0.0

# For each risk category, compute average predicted probabilities
for idx in df.index:
    features_val = X.loc[idx:idx].values
    features_scaled = scaler.transform(features_val)
    probs = model.predict(features_scaled, verbose=0)[0]
    
    # Risk score: weighted combination favoring high-risk probability
    # Range 0-100
    risk_score = (
        probs[0] * 0 +      # Low risk = 0 points
        probs[1] * 50 +     # Moderate risk = 50 points
        probs[2] * 100      # High risk = 100 points
    )
    df.at[idx, 'Risk_Score'] = risk_score

# Compute statistics
risk_thresholds = {
    'low_risk_max': float(df[df['Risk_Category'] == 0]['Risk_Score'].quantile(0.95)),
    'moderate_risk_max': float(df[df['Risk_Category'] == 1]['Risk_Score'].quantile(0.95)),
    'mean_risk_by_category': {
        'low_risk': float(df[df['Risk_Category'] == 0]['Risk_Score'].mean()),
        'moderate_risk': float(df[df['Risk_Category'] == 1]['Risk_Score'].mean()),
        'high_risk': float(df[df['Risk_Category'] == 2]['Risk_Score'].mean())
    }
}

with open('risk_thresholds.json', 'w') as f:
    json.dump(risk_thresholds, f, indent=2)
print("  Risk thresholds saved as risk_thresholds.json")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("\nGenerated Files:")
print("  ✓ model.h5 - Trained neural network model")
print("  ✓ scaler.pkl - Feature scaler for preprocessing")
print("  ✓ model_metadata.json - Feature metadata and ranges")
print("  ✓ risk_thresholds.json - Risk scoring thresholds")
print("\nModel Performance:")
print(f"  Accuracy: {accuracy * 100:.2f}%")
print(f"  Loss: {loss:.4f}")
print("\nReady for deployment in Flask application!")
print("=" * 80)
