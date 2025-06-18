import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# ✅ Sample training data
data = pd.DataFrame({
    'area': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'bathrooms': [1, 2, 2, 3, 3],
    'price': [5000000, 7000000, 8500000, 12000000, 15000000]
})

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# ✅ Save model to file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
