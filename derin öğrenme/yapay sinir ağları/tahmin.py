from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
def optimizasyon(metin):
    stop_words=set(stopwords.words("turkish"))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)
    metin=metin.lower()
    metin=re.sub(r'http\S+', '', metin)
    metin = re.sub('\[[^]]*\]', '', metin)
    metin = (" ").join([word for word in metin.split() if not word in stop_words])
    metin = "".join([char for char in metin if not char in noktalamaIsaretleri])
    return metin



def datasetoptimizasyon(dataset):
    dataset=dataset.dropna()
    stop_words=set(stopwords.words("turkish"))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)
    for ind in dataset.index:
        body = dataset['Body'][ind]
        body = body.lower()
        body = re.sub(r'http\S+', '', body)
        body = re.sub('\[[^]]*\]', '', body)
        body = (" ").join([word for word in body.split() if not word in stop_words])
        body = "".join([char for char in body if not char in noktalamaIsaretleri])
        dataset['Body'][ind] = body
    return dataset



dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/derin öğrenme/yapay sinir ağları/dataset.csv")
dataset=datasetoptimizasyon(dataset)
maxlen=1000
def metindonustur(testData,dataset):
    testData_splited = [testData.split() ]                              
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(testData_splited)
    testData_tok = tokenizer.texts_to_sequences(testData_splited)
    kelime_index = tokenizer.word_index
    kelime_sayi = len(kelime_index) + 1
    testData_tok_pad = pad_sequences(testData_tok, maxlen)
    text_test_tok = tokenizer.texts_to_sequences(dataset.loc[:,"Body"])
    text_test_tok_pad = pad_sequences(text_test_tok, maxlen=maxlen)
    return testData_tok_pad



model=load_model("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/derin öğrenme/yapay sinir ağları/egitilmiş_model.h5")

def tahminEt(inputData):

    #inputData = " " #dataset['Body'][3005]
    inputData = optimizasyon(inputData)
    testData = metindonustur(inputData,dataset)

    print("haber : \n" + inputData)
    pred = model.predict([testData]) 
    print(pred)
    if (pred > 0.95):
        print("Bu haber doğru olarak tahmin edilmiştir")
        return True
    else:
        print("Bu haber sahte olarak tahmin edilmiştir.")
        return False





haber=input("Haber giriniz:")
tahminEt(haber)







