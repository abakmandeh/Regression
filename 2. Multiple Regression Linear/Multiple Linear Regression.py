#Multiple Linear Regression

#y = a_{{0}} + a_{{1}}*x_{{1}}
#
#di mana y adalah variabel dependen, a0 adalah konstanta, a1 adalah koefisien untuk x1, dan x1 adalah variabel independen.
#
#Persamaan MLR: 
#y = a_{{0}} + a_{{1}}*x_{{1}} + a_{{2}}*x_{{2}} + ... + a_{{n}}*x_{{n}}
#

#    Variabel dependen = Profit (data numerik)
#    Variabel independen 1 = Biaya R&D (data numerik)
#    Variabel independen 2 = Biaya administrasi (data numerik)
#    Variabel independen 3 = Biaya marketing (data numerik)
#    Variabel independen 4 = Wilayah perusahaan (data kategori)

# Mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Tampilkan_X = pd.DataFrame(X) #visualisasi X
y = dataset.iloc[:, 4].values
 
# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#karena punya 3 jenis kategori (new york, california, florida)
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
 
# Menghindari jebakan dummy variabel
X = X[:, 1:]
 
# Membagi data menjadi the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Membuat model Multiple Linear Regression dari Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
# Memprediksi hasil Test set
y_pred = regressor.predict(X_test)
 
# Memilih model multiple regresi yang paling baik dengan metode backward propagation
import statsmodels.api as sma
X = sma.add_constant(X)
import statsmodels.api as sm
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#Dengan demikian kita bisa menuliskan fungsi multiple regresinya sebagai berikut:
#Profit = 49030 + 0.8543*BiayaR.D

