import pandas as pd 
df = pd.read_csv("medical_insurance.csv")
print(df.head())
from sklearn.preprocessing import OrdinalEncoder

df['sex'] = df['sex'].map({'male':1, 'female':0})
df['smoker'] = df['smoker'].map({"yes":1,'no':0})
df['region'] = df['region'].map({'southeast':0,"southwest":1,'northwest':2,"northeast":3})
print(df.head())
features = ['age','bmi','children']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[features] = scaler.fit_transform(df[features])

df.head()
from sklearn.model_selection import train_test_split

X = df.drop("charges",axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


lr_model = LinearRegression()

lr_model.fit(X_train,y_train)

y_pred = lr_model.predict(X_test)

print("r2 score :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor()

rf_model.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)

print("r2 score :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
import pickle 

pickle.dump(rf_model,open("rf_model.pkl",'wb'))
pickle.dump(scaler,open("scaler.pkl",'wb'))
import numpy as np 

rf_model.predict(np.array([df.iloc[10,:-1]]))[0]
rf_model.predict(np.array([df.iloc[150,:-1]]))[0]