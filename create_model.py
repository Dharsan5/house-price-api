import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# 1. Load your dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
housing_csv_path = os.path.join(script_dir, 'housing.csv')
df = pd.read_csv(housing_csv_path)

# 2. Clean data (remove rows with missing values)
df.dropna(inplace=True)

# 3. Select features for the app
# We will use 3 simple features for the UI:
# - Median Income (Strongest predictor)
# - Total Rooms
# - House Age
features = ['median_income', 'total_rooms', 'housing_median_age']
target = 'median_house_value'

X = df[features]
y = df[target]

# 4. Train the model
model = LinearRegression()
model.fit(X, y)
print("✅ Model trained successfully on housing.csv!")

# 5. Save the trained model
model_path = os.path.join(script_dir, 'house_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to {model_path}")