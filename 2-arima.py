# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:49:28 2024

@author: Yunus Gümüşsoy

Zaman Serileri - Yolcu Sayısı Tahmin
Time Series - Monthly airline passengers

"""

# !pip install pmdarima
# !pip install arch
# !pip install salesforce-merlion

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
from arch.unitroot import ADF, KPSS
from merlion.models.utils.autosarima_utils import nsdiffs
from pmdarima import auto_arima


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
# 2.Model: Holt Winters Model
# Üstsel düzeltme modelleri

# Datayı train ve test olarak böleceğiz
veri_train=veri_aylık["Yolcu Sayısı"].iloc[:-12]    #son 12 ay hariç tüm datayı eğitim için ayırıyoruz
veri_test=veri_aylık["Yolcu Sayısı"].iloc[-12:]     #son 12 ayı test için ayırıyoruz

trend_tip=["add", "mul"]
seasonal_tip=["add", "mul"]

# en uygun modeli bulmak için döngü tasarlıyoruz
for i in trend_tip:
    for j in seasonal_tip:
        holtwinmodel=ExponentialSmoothing(veri_train, trend=i, seasonal=j, seasonal_periods=12).fit(optimized=True)
        tahmin=holtwinmodel.forecast(12)
        rmse=np.sqrt(mean_squared_error(veri_test, tahmin))
        print("Trend: {} Mevsimsellik: {} RMSE: {}".format(i,j,rmse))

# Trend: add Mevsimsellik: add RMSE: 16.979663509487658
# Trend: add Mevsimsellik: mul RMSE: 15.810478481502829
# Trend: mul Mevsimsellik: add RMSE: 16.435647765201722
# Trend: mul Mevsimsellik: mul RMSE: 25.808050789372796

# Burada RMSE değeri en düşük modeli yani 2. sıradakini seçiyoruz. Trend toplamsal, Mevsimsellik çarpımsal

holtwinmodel=ExponentialSmoothing(veri_train, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
tahmin=holtwinmodel.forecast(12)
rmse=np.sqrt(mean_squared_error(veri_test, tahmin))
print(rmse)

# Tahmin için Kullandığımız HOLT WINTERS Modelinin hata metriği, RMSE: 15.810478481502829


#tahmin edilen datayı görselleştiriyoruz
plt.plot(veri_train, label="Eğitim Verisi")
plt.plot(veri_test, label="Test Verisi")
plt.plot(tahmin, label="Tahmin")
plt.legend()
plt.show()

"""************************************************************************"""

# 3.Model: ARIMA Model
# Üstsel düzeltme modelleri

veri_train=pd.DataFrame(veri_train)
veri_test=pd.DataFrame(veri_test)

# Datamızda yapısal kırılma yok
# Datamızda yaptığımız önceki incelemelerde mevsimsellik ve trend tespit etmiş, serinin durağan olmadığını görmüştük
# Fakat durağanlık testini istatistiksel olarak da görmek isteriz
# Birim kök testleri için Augmented Dickey Fuller (ADF) ve Kwiatkowski-Phillips-Schmidt Shin (KPSS) testleri kullanacağız

adf=ADF(veri_train, trend="ct")
print (adf)
print(adf.regression.summary())

# trend yapımız 0.034, 0.05 den küçük olduğu için anlamlı bir trende sahibiz
# ADF P-value 0.579, 0.05 den büyük olduğu için null hypothesis i reddedemeyiz, seri birim kök barındırıyor, yani durağan değil

# ADF test sonuçlarımızı KPSS testi ile teyit etmek istiyoruz
kpss=KPSS(veri_train, trend="ct")
print(kpss)

# KPSS P-value 0.219, 0.05 den büyük olduğu için null hypothesis i reddedemeyiz, seri durağan
# Durağan olmayan bir seride birim kök tespit etmiş olduk
# Bunun sebepleri; yetersiz gözlem sayısı, trend, ve mevsimsellik olabilir. 
# Ayrıca yapısal kırılmalar da olabilir ama bizim serimizde yapısal kırılma yok

# Mevsimselliği düzeltmek için test yapıyoruz
# Canova-Hansen test
ch=nsdiffs(veri_train["Yolcu Sayısı"], m=12, test="ch")

ocbs=nsdiffs(veri_train["Yolcu Sayısı"], m=12, test="ocbs")

print(ch)
print(ocbs)

# iki testin sonucu da 0 olarak geldi, mevsimselliği ayrıştırıp tekrar test edeceğiz
ayrıs=seasonal_decompose(veri_train["Yolcu Sayısı"], model="mul", period=12, extrapolate_trend="freq")

# modelimiz çarpımsal olduğu için mevsimsel düzeltme/hatayı son değişkende bölerek buluyoruz
veri = pd.concat(
    [ayrıs.observed,
     ayrıs.trend,
     ayrıs.seasonal,
     ayrıs.observed/ayrıs.seasonal], axis=1)

veri.columns=["Orjinal Gözlem", "Trend", "Mevsimsellik", "Mevsimsel Düzeltme"]
# print(veri) #kontrol

adf=ADF(veri["Mevsimsel Düzeltme"], trend="ct")
print (adf)

kpss=KPSS(veri["Mevsimsel Düzeltme"], trend="ct")
print(kpss)

# yeni sonuçlara göre ADF testi serinin birim kök içerdiğini, KPSS testi serinin durağan olmadığını söylüyor, yani çelişki yok

# arima modelimizi kuruyoruz
model=auto_arima(veri_train["Yolcu Sayısı"], trace=False, seasonal=True, m=12)
print(model)

#modelden 12 aylık bir tahmin alıyoruz
tahmin=model.predict(12)
rmse=np.sqrt(mean_squared_error(veri_test["Yolcu Sayısı"], tahmin))
print(rmse)

# Tahmin için Kullandığımız ARIMA Modelinin hata metriği, RMSE: 18.53646857871128


"""************************************************************************"""


# 4. Model: Revize edilmiş ARIMA Model
# Mevsimselliği düzelterek Arima modelimizi kuruyoruz, seasonal ı False yapıyoruz, çünkü kendimiz düzelttik

veri_test["Mevsimsellik"]=veri["Mevsimsellik"].iloc[:12].values
print(veri_test)

model=auto_arima(veri["Mevsimsel Düzeltme"], trace=False, seasonal=False, m=12)
tahmin=model.predict(12)
tahmin=tahmin*veri_test["Mevsimsellik"]
rmse=np.sqrt(mean_squared_error(veri_test["Yolcu Sayısı"], tahmin))
print(rmse)

# Tahmin için Kullandığımız Revize ARIMA Modelinin hata metriği, RMSE: 26.38280395354861

"""************************************************************************"""

# MODEL KARŞILAŞTIRMA
holtwinmodel=ExponentialSmoothing(veri_train, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
model=auto_arima(veri_train["Yolcu Sayısı"], trace=False, seasonal=True, m=12)

tahminhw=holtwinmodel.forecast(12)
tahminar=model.predict(12)

plt.plot(veri_train["Yolcu Sayısı"], label="Eğitim Verisi")
plt.plot(veri_test["Yolcu Sayısı"], label="Test Verisi")
plt.plot(tahminhw, label="Holt Winters Tahmin")
plt.plot(tahminar, label="SARIMA Tahmin")
plt.legend()
plt.show()

