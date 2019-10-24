# Impor library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Impor dataset
dataset = pd.read_csv('Daftar_gaji.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
 
# Membagi data menjadi Training Set dan Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Fitting Simple Linear Regression terhadap Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
# Memprediksi hasil Test Set
y_pred = regressor.predict(X_test)
 
# Visualisasi hasil Training Set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training set)')
plt.xlabel('Tahun bekerja')
plt.ylabel('Gaji')
plt.show()
 
# Visualisasi hasil Test Set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Gaji vs Pengalaman (Test set)')
plt.xlabel('Tahun bekerja')
plt.ylabel('Gaji')
plt.show()
 
from sklearn.metrics import accuracy_score
print('Accuracy score: ', accuracy_score(y_test, y_pred))