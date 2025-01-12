import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lojistik regresyon/bot tespiti/dataset.csv")
dataset.head(12)
dataset.sort_values("Body",inplace=True)
dataset=dataset.drop(columns="B")

dataset.drop_duplicates(subset="Body",keep=False,inplace=True)

def optimizasyon(dataset):
    #non değereleri siler
    dataset=dataset.dropna()
    stop_words=set(stopwords.words('turkish'))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)
    for ind in dataset.index:
        body=dataset["Body"][ind]
        body=body.lower()
        body=re.sub(r'http\S+','',body)
        body=(" ").join([word for word in body.split() if not word in stop_words])
        body="".join([char for char in body if not char in noktalamaIsaretleri])
        dataset['Body'][ind]=body
    return dataset

dataset=optimizasyon(dataset)


yorumlar_makina=dataset[dataset["Label"]==0]
yorumlar_insan=dataset[dataset["Label"]==1]

tfIdf=TfidfVectorizer(binary=False,ngram_range=(1,3))

yorumlar_makina_vec=tfIdf.fit_transform(yorumlar_makina["Body"].tolist())
yorumlar_insan_vec=tfIdf.fit_transform(yorumlar_insan["Body"].tolist())
## tolist komutu body sütunundaki verileri işlenmek üzere listeye dönüştürür


x=dataset.loc[:,"Body"]
y=dataset.loc[:,"Label"]
x_vec=tfIdf.fit_transform(x)
x_eğitim_vec, x_test_vec, y_eğitim, y_test=train_test_split(x_vec,y,test_size=0.2,random_state=0)
print(x_vec)
# lojisyikregresyon=LogisticRegression()

# lojisyikregresyon.fit(x_eğitim_vec, y_eğitim)

# pickle.dump(lojisyikregresyon,open("egitilmiş_model","wb"))
# print("Lojiksik regresyon modeli eğitildi ve kaydedildi")

# pickle.dump(tfIdf,open("vektörleştirici","wb"))
# print("TF-IDF vektörleştirici kaydedildi")




