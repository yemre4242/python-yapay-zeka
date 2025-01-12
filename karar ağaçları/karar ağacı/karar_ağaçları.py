import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler####BU kütüphane bazı değerlerin ağırlığı aşırı etkileyerek baskın hale gelmesini engeller örneğin bu 
#projede alyuvar sayısı büyük değerler alacağı için ağırlığı çok etkileyecektir.Bu kütüphane içindeki fonksiyonlar bunu engellemek üzerinedir.

dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/karar ağaçları/karar ağacı/veriseti.csv")

x=dataset.iloc[:,[1,2,3,4]].values####Burası çok kritik çünkü normal pandas dizisi olan değerler .values komutu ile sadece değerlerin alınmasını sağlayarak 
#liste haline getirildi ve kolaylıkla liste içerisinde işlem yapılabildi
etiket=dataset.iloc[:,5].values
for e in x:
    if e[0]=="Erkek":
        e[0]=1
    elif e[0]=="Kadın":
        e[0]=0

scaler=StandardScaler()
scaler.fit(x)
parametre=scaler.transform(x)
classifer=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifer.fit(parametre,etiket)


print("Hastanın cinsiyetini giriniz:(Kadın/Erkek)")
cinsiyet=input()

if cinsiyet=="Kadın":
    cinsiyet=0
elif cinsiyet=="Erkek":
    cinsiyet=1

print("Hastanın yaşını giriniz:")
yaş=int(input())

print("Hastanın alyuvar sayısını giriniz:")
akyuvar_sayisi=int(input())

print("Hastanın kornik rahatsızlığı var mı:(evet/hayır)")
kronik_rahatsızlık=input()

if kronik_rahatsızlık=="evet":
    kronik_rahatsızlık=1
elif kronik_rahatsızlık=="hayır":
    kronik_rahatsızlık=0


inputdata=np.array([cinsiyet,yaş,akyuvar_sayisi,kronik_rahatsızlık]).reshape(1,-1)
testvektör=scaler.transform(inputdata)

prediction_result=classifer.predict(testvektör)

if prediction_result == 1:
	print("Tahmin edilen durum: ağır seyir beklenilir")
if prediction_result == 0:
	print("Tahmin edilen durum: ayakta iyileşme beklenilir")








