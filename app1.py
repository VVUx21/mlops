import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore

dataset = pd.read_csv("car data .csv")

# Important..
X = dataset.drop(['Selling_Price', 'Car_Name'], axis=1).values
y = dataset['Selling_Price'].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3, 4, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X, y)

joblib.dump(model, "car_price_model.pkl")
print("Model saved successfully!")
# This script trains a Random Forest Regressor model on the car dataset and saves the model to a file named "car_price_model.pkl".