from sklearn.linear_model import LogisticRegression
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
dataset=pd.read_csv("C:/Users/yuemr/Downloads/student_performance_prediction.csv")


dataset=dataset.dropna()
dataset=dataset.drop(columns="Participation in Extracurricular Activities")
dataset=dataset.drop(columns="Student ID")

dataset=dataset.drop(columns="Attendance Rate")

dataset.drop_duplicates(subset="Study Hours per Week",keep="first",inplace=True)

dataset=dataset[dataset["Study Hours per Week"]>0.0]
tfIdf=TfidfVectorizer(binary=False,ngram_range=(1,3))
başarili=dataset[dataset["Passed"]=="Yes"]
başarisiz=dataset[dataset["Passed"]=="No"]
x = dataset.loc[:, ["Study Hours per Week","Previous Grades"]]
y=dataset.loc[:,"Passed"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

smote = SMOTE(random_state=0)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

model=LogisticRegression()

model.fit(x_train_resampled, y_train_resampled)

y_pred=model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluk Oranı: {accuracy}")
yeni_öğrenci=pd.DataFrame([[-8.5,75]], columns=["Study Hours per Week","Previous Grades"])

tahmin=model.predict_proba(yeni_öğrenci)
print(model.predict(yeni_öğrenci))
print(tahmin)

