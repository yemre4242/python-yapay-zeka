import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression



dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lineer regresyon/linear_regression_dataset.csv")
x=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values
lineer_regresyon_model=LinearRegression()
yıllar=[[13],[15],[24]]
eğitilmiş=lineer_regresyon_model.fit(x,y)
tahmin_sonucu=eğitilmiş.predict(yıllar)


plt.scatter(x,y,color='blue')
plt.scatter(yıllar,tahmin_sonucu,color="red")
plt.plot(x,lineer_regresyon_model.predict(x),color='green')
plt.title("maaş tahmini")
plt.xlabel("yıl")
plt.ylabel("maaş")
plt.grid(True)
plt.show()


