import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import gensim
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

dataset=pd.read_csv("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/derin öğrenme/yapay sinir ağları/dataset.csv")
dataset.head()
dataset.sort_values("Body",inplace=True)
dataset.drop_duplicates(subset="Body",keep=False,inplace=True)


def optimizasyon(dataset):
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


def trASCIcevirici(metin):
    translationTable=str.maketrans("ğĞıİöÖüÜşŞçÇ","gGiIoOuUsScC")
    metin=metin.translate(translationTable)
    return metin

dataset=optimizasyon(dataset)

x=dataset.loc[:,"Body"]
y=dataset.loc[:,"Label"]

x_eğitim,x_test,y_eğitim,y_test=train_test_split(x,y,test_size=0.2,random_state=28)
x_eğitim_dizi=[metin.split() for metin in x_eğitim]

maxmesafe=2
minfrekans=1
vektor_boyutu=200
w2v_model=gensim.models.Word2Vec(sentences=x_eğitim_dizi,vector_size=vektor_boyutu,window=maxmesafe,min_count=minfrekans)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_eğitim_dizi)
x_eğitim_tok=tokenizer.texts_to_sequences(x_eğitim_dizi)
kelime_index=tokenizer.word_index
maxlen=1000
x_eğitim_tok_pad=pad_sequences(x_eğitim_tok,maxlen=maxlen)
kelime_sayi=len(kelime_index)+1


matris=np.zeros((kelime_sayi,vektor_boyutu))
for kelime,i in kelime_index.items():
    matris[i]=w2v_model.wv[kelime]

model=Sequential()
 



model.add(Embedding(matris.shape[0],output_dim=matris.shape[1],weights=[matris],input_length=maxlen,trainable=False))



model.add(LSTM(units=32))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])

model.summary()

model.fit(x_eğitim_tok_pad,y_eğitim,validation_split=0.2,epochs=30,batch_size=32,verbose=1)
model.save("egitilmiş_model.h5")
print("model eğitildi ve kaydedildi!")