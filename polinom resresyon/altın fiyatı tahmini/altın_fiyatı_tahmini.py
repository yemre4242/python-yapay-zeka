import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/polinom resresyon/altın fiyatı tahmini\Gold Price Prediction.csv")
dataset.drop(columns=["Price 2 Days Prior","Price 1 Day Prior","Price Change Tomorrow","200 Day Moving Average","Treasury Par Yield Two Year","Treasury Par Yield Curve Rates (10 Yr)"])

dataset=dataset.dropna()




x=dataset.loc[:,["Price Today","Std Dev 10","Twenty Moving Average","Fifty Day Moving Average","Volume ","Treasury Par Yield Month","Monthly Inflation Rate","EFFR Rate","DXY","SP Open","VIX","Crude"]]
y=dataset.loc[:,["Price Tomorrow"]]



poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)

liner_reg=LinearRegression()

liner_reg.fit(x_poly,y)
y_pred=liner_reg.predict(x_poly)

fig2,(bx0,bx1,bx2)=plt.subplots(nrows=3,figsize=(6,10))
bx2.plot(y,label="gerçek",c="b")
bx2.plot(y_pred,label="tahmin",c="r")
bx2.set_title("gerçek - tahminin altın fiyatları ")
plt.legend()
plt.tight_layout()
plt.show()







