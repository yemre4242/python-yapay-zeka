import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/karar ağaçları/Rastgele orman algoritması/rastsal.csv")

parametre=dataset.iloc[:,1:2].values
hedef=dataset.iloc[:,2].values

kararAgaci=RandomForestRegressor(n_estimators=20,random_state=0)

kararAgaci.fit(parametre,hedef)

print("çalışan kişinin deneyimini yıl cinsinden giriniz:")
deneyim=input()

tahmin=kararAgaci.predict(np.array([deneyim]).reshape(1,1))

print(str(deneyim)+" yıl deneyimi olan birisinin tahmini maaşı :"+ str(tahmin[0])+"TL olur.")


