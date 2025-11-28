import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load and clean the source dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
housing_csv_path = os.path.join(script_dir, 'housing.csv')
df = pd.read_csv(housing_csv_path)

# Focus only on the columns the app uses so missing data does not skew the fit
target = 'median_house_value'
numeric_features = ['median_income', 'total_rooms', 'housing_median_age', 'longitude', 'latitude']
categorical_features = ['ocean_proximity']
features = numeric_features + categorical_features
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Encode ocean proximity while leaving numeric features untouched
preprocessor = ColumnTransformer(
    transformers=[
        ('ocean', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough',
)

# Train a simple linear regression inside a pipeline so preprocessing ships with the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()),
])
model.fit(X, y)
print("✅ Model trained successfully on housing.csv with location and ocean proximity features!")

# Persist the fitted pipeline to disk for reuse in the API
model_path = os.path.join(script_dir, 'house_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to {model_path}")