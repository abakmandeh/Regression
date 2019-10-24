#Regresi polinomial merupakan regresi di mana fungsinya adalah kuadratik. Perbedaan persamaannya bisa kita lihat sebagai berikut:
#
#Simple linear –> 
#y = a_{{0}} + a_{{1}}*x_{{1}}
#
#Multilinear –>  
#y = a_{{0}} + a_{{1}}*x_{{1}} + a_{{2}}*x_{{2}} + ... + a_{{n}}*x_{{n}}
#
#Polinomial –> 
#y=a_{0} + a_{1}*x_{1} + a_{2}*{x_{{1}}}^{2} + ... + a_{n}*{x_{{n}}}^{n}
 

#multilinear.
#hubungan antara 1 variabel dependen dengan banyak variabel independen,
#
#Kapan simple atau poli?
#hubungan antara 1 variabel dependen dengan 1 variabel independen
#
#* Jika menggunakan simple linear sudah fit, maka cukup menggunakan model ini saja, 
#namun jika tidak dan fungsinya tampak seperti fungsi polinomial (fungsi kuadratik) 
#maka kita coba dekati dengan metode polinomial. 
#
#Jika menggunakan simple dan polinomial tidak juga fit, 
#maka hubungan antara keduanya bukanlah linear, 
#sehingga harus menggunakan algoritma regresi non linear misal seperti SVR (support vector regression) 

#
#STUDI KASUS
#
#Dalam pembelajaran kali ini, kita ingin mencari solusi dari proses perekrutan sebuah perusahaan. Perusahaan ini sedang merekrut seorang calon pegawai baru. Namun, bagian HRD perusahaan ini kebingungan, berapa gaji yang harus ia berikan, sesuai dengan level di mana calon pegawai baru ini masuk. Tentunya akan ada proses negosiasi antara HRD dengan calon pegawai baru ini tentang jumlah gaji yang pantas diterima pegawai tersebut.
#
#Calon pegawai ini mengaku bahwa sebelumnya ia telah berada di posisi Region Manager dengan pengalaman bekerja 20 tahun lebih dengan gaji hampir 160K dollar per tahun. Ia meminta perusahaan baru ini untuk memberikan ia gaji lebih dari 160K dollar per tahun.
#
#Untuk menyelidiki apakah calon pegawai ini benar-benar digaji sebanyak 160K dollar/tahun, maka bagian HRD membandingkan data gaji perusahaan tempat calon pegawai ini bekerja sebelumnya (kebetulan perusahaan memiliki daftar gajinya) dengan pengakuannya.
#
#Data yang dimiliki adalah daftar antara gaji dan level di perusahaan tersebut. Bagian HRD ingin mencari hubungan antara gaji yang didapat dengan level (tingkatan jabatan) di perusahaan calon pekerja tadi bekerja sebelumnya.
#
#Hasil penelitian awal, calon pegawai ini layak masuk di level 6.5 (antara region manager dan partner).
#
#Berikut variabel yang kita miliki:
#
#    Variabel dependen : Gaji (dalam dollar per tahun)
#    Variabel independen : level (tingkatan jabatan)

# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('Posisi_gaji.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
 
# Fitting Linear Regression ke dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
 
# Fitting Polynomial Regression ke dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)  ## nantinya degree diganti menjadi 4
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
 
# Visualisasi hasil regresi sederhana
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Sesuai atau tidak (Linear Regression)')
plt.xlabel('Level posisi')
plt.ylabel('Gaji')
plt.show()
 
# Visualisasi hasil regresi polynomial
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Sesuai atau tidak (Polynomial Regression)')
plt.xlabel('Level posisi')
plt.ylabel('Gaji')
plt.show()
 
# Memprediksi hasil dengan regresi sederhana
lin_reg.predict(6.5)
 
# Memprediksi hasil dengan regresi polynomial
lin_reg_2.predict(poly_reg.fit_transform(6.5))