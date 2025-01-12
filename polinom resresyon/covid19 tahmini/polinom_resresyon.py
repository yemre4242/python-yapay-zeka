import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import date
dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/polinom resresyon/covid19.csv")

dataset=dataset.drop(columns=["ortalama_filyasyon_suresi","yatak_doluluk_orani","ventilator_doluluk_orani","ortalama_temasli_tespit_suresi","filyasyon_orani","gunluk_test","gunluk_iyilesen","toplam_test","toplam_hasta","toplam_vefat","toplam_iyilesen","toplam_yogun_bakim","toplam_entube","hastalarda_zaturre_oran"])

dataset=dataset.dropna()
dataset=dataset[::-1].reset_index()

X=dataset.loc[:, ["tarih","gunluk_vaka","gunluk_hasta","agir_hasta_sayisi","eriskin_yogun_bakim_doluluk_orani"]]
y=dataset.loc[:,["gunluk_vefat"]]


l=len(X.index)
for ind in X.index:
    X["tarih"][ind]=l-ind
    X["gunluk_vaka"][ind]=float(str(X["gunluk_vaka"][ind]).replace(".",""))
    X["gunluk_hasta"][ind]=float(str(X["gunluk_hasta"][ind]).replace(".",""))
    X["agir_hasta_sayisi"][ind]=float(str(X["agir_hasta_sayisi"][ind]).replace(".",""))
    X["eriskin_yogun_bakim_doluluk_orani"][ind]=float(str(X["eriskin_yogun_bakim_doluluk_orani"][ind]).replace(",","."))

poly_reg=PolynomialFeatures(degree=3)
y_len=len(y)
x_poly=poly_reg.fit_transform(X)

lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)

y_pred=lin_reg.predict(x_poly)
# print(y_pred)
# print(y_len,"adet gün için tahmin yapılmıştır")

# for i in range(y_len):
#     print("|gerçek :"+str(y["gunluk_vefat"][i])+"| Tahmin ->"+str(y_pred[i]))

fig2,(bx0,bx1,bx2)=plt.subplots(nrows=3,figsize=(6,10))
bx2.plot(y,label="gerçek",c="b")
bx2.plot(y_pred,label="tahmin",c="r")
bx2.set_title("gerçek - tahminin ölüm sayıları")
plt.legend()
plt.tight_layout()
plt.show()
# grafik_eriskin_yogun_bakim_doluluk_orani=X.loc[:,["eriskin_yogun_bakim_doluluk_orani"]]
# bx1.plot(grafik_eriskin_yogun_bakim_doluluk_orani,label="yoğun bakım",c="r")
# bx1.set_title("yoğun bakım doluluk oranı (%)")

# grafik_agir_hasta=X.loc[:,["agir_hasta_sayisi"]][::-1]
# bx0.plot(grafik_agir_hasta,label="ağır hasta",c="r")
# bx0.set_title("ağır hasta sayısı")



# fig,(ax0,ax1,ax2)=plt.subplots(nrows=3,figsize=(6,10))
# ax0.scatter(X.loc[:,["tarih"]], y,s=10,c='b',marker="o")
# ax0.set_title("zaman - ölüm")

# grafik_gunluk_vaka=X.loc[:,["gunluk_vaka"]]
# ax1.plot(grafik_gunluk_vaka,label="günlük vaka",c="r")
# ax1.set_title("gunluk vaka sayısı")

# grafik_günlük_hasta_sayisi=X.loc[:,["gunluk_hasta"]]
# ax2.plot(grafik_günlük_hasta_sayisi,label="günlük hasta",c="r")
# ax2.set_title("günlük hasta sayısı")



# def tariholuştur(yenitarih):
    
#     dizi=yenitarih.split(".")
#     yenitarih=date(int(dizi[2]),int(dizi[1]),int(dizi[0]))
#     başlangıçtarihi=date(2021,4,16)
#     fark=yenitarih-başlangıçtarihi
#     return fark.days


# tarih="03.09.2020"
# fix_tarih=tariholuştur(tarih)   
# vakasayisi=40444
# hastasayisi=2728
# agirhastasayisi=3558
# yoğunbakımoranı=50
# tahmin_edilecek_veri=np.array([fix_tarih,vakasayisi,hastasayisi,agirhastasayisi,yoğunbakımoranı]).reshape(1,-1)

# tahmin_edilecek_veri=poly_reg.fit_transform(tahmin_edilecek_veri)
# tahmin=lin_reg.predict(tahmin_edilecek_veri)
# # print("'"+ tarih+"' tarihi için covid 19 kaynaklı vefat sayısı : "+str(tahmi
# print(tahmin)



