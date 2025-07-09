import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('housing.csv')

# Features & target
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
