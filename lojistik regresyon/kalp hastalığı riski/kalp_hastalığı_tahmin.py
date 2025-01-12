import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


regresyon=pickle.load(open("C:/Users/yuemr\OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lojistik regresyon/kalp hastalığı riski/kalp hastalığı veriseti","rb"))
tahmin=pd.DataFrame([[1,45,1,45,0,0,180,105,84,195,120,78]],columns=["male","age","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","totChol","sysBP","diaBP","BMI","heartRate","glucose"])
tahmin_sonucu=regresyon.predict(tahmin)
print(tahmin_sonucu)
print(regresyon.predict_proba(tahmin))
