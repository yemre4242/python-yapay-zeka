import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import accuracy_score

def optimizasyon(metin):
    stop_words=set(stopwords.words("turkish"))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)
    body=metin
    body=body.lower()
    body=re.sub(r'http\S+','',body)
    body=re.sub('!\[[^]]*\}','',body)
    body=(" ").join([word for word in body.split() if not word in stop_words])
    body="".join([char for char in body if not char in noktalamaIsaretleri])
    return body

tahmin_edilecek_metin=input("Sınıflandırmak üzere bir metin giriniz:")
tahmin_edilecek_metin=optimizasyon(tahmin_edilecek_metin)

tf_idf=pickle.load(open("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lojistik regresyon/vektörleştirici","rb"))

tahmin_edilecek_metin_vec=tf_idf.transform([tahmin_edilecek_metin])

lojistikregresyon=pickle.load(open("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/makina öğrenmesinde matematik modellemesi/lojistik regresyon/egitilmiş_model","rb"))

tahmin_sonucu=lojistikregresyon.predict(tahmin_edilecek_metin_vec)
print(tahmin_sonucu)




