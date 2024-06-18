# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:07:56 2024

@author: Yunus
"""

#pip install evds --upgrade

# TCMB EVDS Sisteminden Otomatik Veri Çekim
import evds as ev
import warnings 
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt


# API anahtarı kişiye özel olduğu için dosyaya kaydedip dosyadan çekiyorum, ama direkt yazılıp da çekilebilir
# evds = e.evdsAPI("API KEY")

with open ("C:/Users/yunus/OneDrive/Masaüstü/evds.txt") as dosya:
    api=dosya.read()
    
    
evds = ev.evdsAPI(api) 

# VAR modeli için veri topluyorum

# EVDS den enflasyon verisini çekiyorum, datanın değişkenlerini sitesinden bakmıştım, farklı değişkenler için farklı kodlar gerekir
enf=evds.get_data(["TP.FG.J0", "TP.TUFE1YI.T1"], startdate="01-01-2010", enddate="01-12-2024")
print(enf) #kontrol

# EVDS den petrol fiyatlarını çekiyorum, datalar uyumlu olması lazım, enflasyon aylık gelirken petrol fiyatları farklı, o yüzden frekansı değiştiriyorum
pf=evds.get_data(["TP.BRENTPETROL.EUBP"], startdate="01-01-2010", enddate="01-12-2024", frequency=5)
print(pf) #kontrol

# EVDS den kur datalarını çekiyorum, aylık frekansta olduğu için ortalama alıyorum
doviz=evds.get_data(["TP.DK.USD.A.YTL"], startdate="01-01-2010", enddate="01-12-2024", frequency=5, aggregation_types="avg")
print(doviz) #kontrol

# EVDS faiz datalarını çekiyorum, vadeli tl mevduat faizi
faiz=evds.get_data(["TP.TRY.MT02"], startdate="01-01-2010", enddate="01-12-2023", frequency=5, aggregation_types="avg")
print(faiz) #kontrol

# tüm dataları tek bir data frame de topluyorum
veri = pd.concat([enf, pf, doviz, faiz], axis = 1)
veri = pd.DataFrame(veri)
print (veri) #kontrol

# her datanın tarih sütunu var, tekrarı engellemek için tarih sütunlarını siliyorum
veri.drop("Tarih", axis=1, inplace=True)
print(veri) #kontrol

# nihai datamızdaki sütun isimlerini değiştiriyorum
veri.rename(columns={"TP_FG_J0":"Tüfe", "TP_TUFE1YI_T1":"Üfe", 
                     "TP_BRENTPETROL_EUBP":"Petrol", "TP_DK_USD_A_YTL":"DolarTL", 
                     "TP_TRY_MT02":"MF"}, inplace = True)

# sildiğim tarih sütunları yerine tarih sütununu indeks olarak ekliyorum
tarih=pd.date_range("01.01.2010", periods=len(veri), freq="M")
veri["Tarih"]=tarih
veri.set_index("Tarih", inplace=True)
print(veri)

# grafik
fig, axes=plt.subplots(3,2)
axes_flat=axes.flatten()

for i, col in enumerate(veri.columns):
    veri[col].plot(ax=axes_flat[i], title=col)
    
plt.tight_layout()
plt.show()

# datayı her seferinde evds den çekmek zorunda kalmamak için excel e çekiyorum
veri.to_excel("C:/Users/yunus/OneDrive/Masaüstü/veri.xlsx")

