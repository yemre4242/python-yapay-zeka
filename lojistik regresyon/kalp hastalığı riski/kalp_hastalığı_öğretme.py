import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset=pd.read_csv(open("C:/Users/yuemr\OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lojistik regresyon/kalp hastalığı riski/framingham.csv"))
print(dataset)
dataset=dataset.dropna()
hasta=dataset[dataset["TenYearCHD"]==1]
sağlikli=dataset[dataset["TenYearCHD"]==0]

x=dataset.loc[:,["male","age","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]

y=dataset.loc[:,"TenYearCHD"]
x_eğitim, x_test, y_eğitim, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lojistik_resgresyon=LogisticRegression(max_iter=4000)

lojistik_resgresyon.fit(x_eğitim,y_eğitim)

y_pred=lojistik_resgresyon.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluk Oranı: {accuracy}")

pickle.dump(lojistik_resgresyon,open("C:/Users/yuemr\OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lojistik regresyon/kalp hastalığı riski/kalp hastalığı veriseti","wb"))
print("Lojiksik regresyon modeli eğitildi ve kaydedildi")




