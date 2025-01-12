import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/veriseti.csv")
x=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values

lineer_regresyon_model=LinearRegression()

lineer_regresyon_model.fit(x,y)

tahminedilecekyillar=[[18],[19],[20],[21]]
tahmin_sonucları=lineer_regresyon_model.predict(tahminedilecekyillar)
print(str(tahminedilecekyillar)+"için sırasıyla tahmin sonuçları:"+str(tahmin_sonucları))

plt.scatter(x,y,color='blue')
plt.scatter(tahminedilecekyillar,tahmin_sonucları,color="red")
plt.plot(x,lineer_regresyon_model.predict(x),color='green')
plt.title("emisyon salınım tahmini")
plt.xlabel("yıl")
plt.ylabel("emisyon salınımı")
plt.show()