# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:49:28 2024

@author: Yunus Gümüşsoy

Zaman Serileri - Yolcu Sayısı Tahmin
Time Series - Monthly airline passengers

"""

# !pip install pmdarima

from pmdarima.datasets import load_airpassengers    
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf, month_plot, quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# hata olmayan uyarıları kapatmak için
import warnings
warnings.filterwarnings("ignore")

# yolcu sayılarını tam sayı olarak almak için dtype=int ekliyoruz
veri_aylık = pd.DataFrame(load_airpassengers(as_series = True), columns=["Yolcu Sayısı"], dtype=int)
# print(veri_aylık)

# dataya tarih ekliyoruz, örnek çalışma olduğu için istediğimiz tarihi yazabiliriz, aylık frekansta almak için M-Month 
tarih=pd.date_range("01-01-1995", periods=len(veri_aylık), freq="M")

# yukarıda oluşturduğumuz tarih indexini dataya sütun olarak ekliyoruz
veri_aylık["Tarih"]=tarih

veri_aylık.set_index("Tarih", inplace=True)
# print(veri_aylık)

# eksik gözlem kontrolü için
# print(veri_aylık.isnull().sum())

# datayı farklı frekanslara çevirmek için
veri_ceyrek = veri_aylık.resample("Q").sum()    #Quarter
veri_yıllık = veri_aylık.resample("Y").sum()    #Year

# print(veri_aylık)
# print(veri_ceyrek)
# print(veri_yıllık)

# veri görselleştirme
fix, ax=plt.subplots(3,1)
ax[0].plot(veri_aylık, label="Aylık")
ax[0].legend(loc="upper left")
ax[1].plot(veri_ceyrek, label="Çeyrek")
ax[1].legend(loc="upper left")
ax[2].plot(veri_yıllık, label="Yıllık")
ax[2].legend(loc="upper left")
plt.show()

# Grafiği incelediğimizde aylık, çeyreylik ve yıllık olarak yükselen bir trend görüyoruz
# Fakat aylık ve çeyreklik grafiklerde belirgin olan mevsimselliğin yıllık grafikte azaldığını görüyoruz
# Bu normal bir sonuç, eğer mevsimsellik araştırıyorsak aylık ve çeyreklik verilere / grafiklere bakmak daha doğru olacaktır

# datanın betimsel analizi, transpose() sonuçları yatayda göstermek için
ist1= veri_aylık.describe().transpose()
# print (ist1) #kontrol
ist2= veri_ceyrek.describe().transpose()
ist3= veri_yıllık.describe().transpose()

# aylık, çeyreklik ve yıllık betimsel analizleri birleştirmek için
betist = pd.concat([ist1, ist2, ist3])

# Data frame yapısına indeks ekleyeceğiz
indeksler = ["Aylık Yolcu Sayısı", "Çeyreklik Yolcu Sayısı", "Yıllık Yolcu Sayısı"]

# indeksleri betimsel analiz sonuçlarımıza ekliyoruz
betist.index = indeksler
print(betist)

# datamızdan aylık ve yıllık verileri alıyoruz
veri_aylık["Ay"]=veri_aylık.index.month
veri_aylık["Yıl"]=veri_aylık.index.year
# print(veri_aylık) #kontrol

matris=pd.pivot_table(veri_aylık, values="Yolcu Sayısı", index="Yıl", columns="Ay")
renk=sns.color_palette("Blues", as_cmap=True)
sns.heatmap(matris, annot=True, fmt="g", cmap=renk) #annot kutular içinde rakam olması, fmt de rakamların okunaklı olması için
plt.show()

# Zaman Serileri çalıştığımız için çeyreklik ve aylık data üzerine yoğunlaşacağız
# çeyreklik datamızdan aylık ve yıllık verileri alıyoruz
veri_ceyrek["Ay"]=veri_ceyrek.index.month
veri_ceyrek["Yıl"]=veri_ceyrek.index.year

# veri görselleştirme
fix, ax=plt.subplots(2,1)
ax[0].plot(veri_aylık["Yolcu Sayısı"], label="Aylık")
ax[0].legend(loc="upper left")
ax[1].plot(veri_ceyrek["Yolcu Sayısı"], label="Çeyrek")
ax[1].legend(loc="upper left")
plt.show()


# Box-Cox dönüşümü, çarpık veriler üzerinde oldukça düzeltici etkilesi olduğu bilinen istatistiksel bir tekniktir. 
# Fonksiyon, normal olmayan dağılımı normal dağılıma dönüştürme türünü lambda (λ) parametresine göre belirlemektedir. 
# Box-Cox için lambda (λ) parametresi -5 <λ <5 aralığına sahiptir. 

#box-cox dönüşüm yapısı tanımlıyoruz, lm = lambda değeri
veri_aylık_boxcox,lm= boxcox(veri_aylık["Yolcu Sayısı"])
veri_ceyrek_boxcox,lm2= boxcox(veri_ceyrek["Yolcu Sayısı"])

#tanımladığımız boxcox değerlerini datalarımıza sütun olarak ekliyoruz
veri_aylık["Box Cox"]=veri_aylık_boxcox
veri_ceyrek["Box Cox"]=veri_ceyrek_boxcox
print(veri_aylık)
print(veri_ceyrek)

# boxcox dönüşümünü içeren veri görselleştirme
fix, ax=plt.subplots(2,2)
ax[0,0].plot(veri_aylık["Yolcu Sayısı"], label="Aylık")
ax[0,0].legend(loc="upper left")
ax[0,1].plot(veri_aylık["Box Cox"], label="Aylık Box Cox")
ax[0,1].legend(loc="upper left")

ax[1,0].plot(veri_ceyrek["Yolcu Sayısı"], label="Çeyreklik")
ax[1,0].legend(loc="upper left")
ax[1,1].plot(veri_ceyrek["Box Cox"], label="Çeyreklik Box Cox")
ax[1,1].legend(loc="upper left")
plt.show()

# otokorelasyon kontrolü için veri görselleştirme
fix, ax=plt.subplots(2,2)
plot_acf(veri_aylık["Yolcu Sayısı"], lags=25, ax=ax[0,0], zero=False)
ax[0,0].set_title("Aylık Otokorelasyon")

plot_acf(veri_aylık["Box Cox"], lags=25, ax=ax[0,1], zero=False)
ax[0,1].set_title("Aylık Box Cox Otokorelasyon")

plot_acf(veri_ceyrek["Yolcu Sayısı"], lags=25, ax=ax[1,0], zero=False)
ax[1,0].set_title("Çeyreklik Otokorelasyon")

plot_acf(veri_ceyrek["Box Cox"], lags=25, ax=ax[1,1], zero=False)
ax[1,1].set_title("Çeyreklik Box Cox Otokorelasyon")

plt.show()

# mevsimsellik kontrolü için veri görselleştirme
fix, ax=plt.subplots(2,2)
month_plot(veri_aylık["Yolcu Sayısı"], ax=ax[0,0])
ax[0,0].set_title("Aylık Mevsimsellik")

month_plot(veri_aylık["Box Cox"], ax=ax[0,1])
ax[0,1].set_title("Aylık Box Cox Mevsimsellik")

quarter_plot(veri_ceyrek["Yolcu Sayısı"], ax=ax[1,0])
ax[1,0].set_title("Çeyreklik Mevsimsellik")

quarter_plot(veri_ceyrek["Box Cox"], ax=ax[1,1])
ax[1,1].set_title("Çeyreklik Box Cox Mevsimsellik")

plt.show()

# mevsimsellik ayrımı
# Ham verilerde değişen varyansı görmek için çarpımsal model olarak ayırıyoruz, aylık olduğu için period 12
a=seasonal_decompose(veri_aylık["Yolcu Sayısı"], model="mul", period=12)
a.plot()

# Boxcox dönüşümünden dolayı varyanslar normalleştiği içn toplamsal model kullanıyoruz
b=seasonal_decompose(veri_aylık["Box Cox"], model="add", period=12)
b.plot()

# çeyrek olduğu için period 4
c=seasonal_decompose(veri_ceyrek["Yolcu Sayısı"], model="mul", period=4)
c.plot()

d=seasonal_decompose(veri_ceyrek["Box Cox"], model="add", period=4)
d.plot()

plt.show()
# Grafikleri incelediğimizde çarpımsal modellerde hataların 1 civarında dağıldığını görürken, toplamsal modellerde 0 civarında dağıldığını görüyoruz


# Tahmin
# 1.Model: Ayrıştırma Modeli
# Datayı train ve test olarak böleceğiz
veri_train=veri_aylık["Yolcu Sayısı"].iloc[:-12]    #son 12 ay hariç tüm datayı eğitim için ayırıyoruz
veri_test=veri_aylık["Yolcu Sayısı"].iloc[-12:]     #son 12 ayı test için ayırıyoruz
# print(veri_test) #kontrol

# eğitim için ayırdığımız datayı mevsimsellikten ayrıştırmak için hazırlıyoruz
ayrıs=seasonal_decompose(veri_train, model="mul", period=12, extrapolate_trend="freq")

# modelimiz çarpımsal olduğu için mevsimsel düzeltme/hatayı son değişkende bölerek buluyoruz
veri = pd.concat(
    [ayrıs.observed,
     ayrıs.trend,
     ayrıs.seasonal,
     ayrıs.observed/ayrıs.seasonal], axis=1)

veri.columns=["Orjinal Gözlem", "Trend", "Mevsimsellik", "Mevsimsel Düzeltme"]
#print(veri) #kontrol

#elde ettiğimiz yeni sütunlardaki verileri indeksleyeceğiz
indeks=np.arange(1,len(veri)+1)
#print(indeks) #kontrol

#dataya T isimli bir sütun ekleyip indeksleri ekliyoruz
veri["T"]=indeks
#print(veri) #kontrol

x=sm.add_constant(veri["T"])
y=veri["Mevsimsel Düzeltme"]
model=sm.OLS(y,x).fit()
print(model.summary())

# R-squared: 0.977
# Datayı mevsimsellikten arındırıp, verilerin geri kalanını mevsimsel düzeltme isimli bir veride topladığımız model başarılı oldu.

# print(veri["Mevsimsellik"].iloc[:13]) #kontrol

# mevsimsellik değişkenini test verisine de ekliyoruz
veri_test=pd.DataFrame(veri_test)
veri_test["Mevsimsellik"]=veri["Mevsimsellik"].iloc[:12].values
#print(veri_test) #kontrol

# test verimizde değerler için 12 adet indeks yapıyoruz, ilk datayı başta indeksleseydik bunlara gerek kalmayacaktı
girdi=np.arange(len(veri.index)+1, len(veri.index)+13)

#print(indeks) #kontrol
#print(girdi) #kontrol

tahmin=model.predict(sm.add_constant(girdi))

# Datamızdan mevsimselliği ayrıştırırken Orijinal veriyi Mevsimselliğe bölmüştük
# şimdi mevsimsellikten ayırarak elde ettiğimiz tahmini mevsimsellik ile çarpıyoruz
veri_test["Tahmin"]=veri_test["Mevsimsellik"]*tahmin
print(veri_test)

#tahmin edilen datayı görselleştiriyoruz
plt.plot(veri["Orjinal Gözlem"], label="Eğitim Verisi")
plt.plot(veri_test["Yolcu Sayısı"], label="Test Verisi")
plt.plot(veri_test["Tahmin"], label="Tahmin")
plt.legend()
plt.show()

# Tahmin için Kullandığımız Ayrıştırma Modelinin hata metriği olarak root-mean-square error (RMSE) kullanıyoruz
print(np.sqrt(mean_squared_error(veri_test["Yolcu Sayısı"], veri_test["Tahmin"])))

# Tahmin için Kullandığımız Ayrıştırma Modelinin hata metriği, RMSE: 38.908370941910384






