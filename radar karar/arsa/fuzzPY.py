import numpy as np

def uçgen(x,abc):
    assert len(abc)==3,'Başlangıç, tepe ve bitiş noktaları verilmeldir'
    a,b,c=np.r_[abc]
    x=np.array(x)
    assert a<=b and b<=c
    y =np.zeros(len(x))
    if a != b:
        idx=np.nonzero(np.logical_and(a < x,x < b))[0]
        y[idx]=(x[idx]-a)/float(b-a)
    if b!=c:
        idx=np.nonzero(np.logical_and(b < x,x < c))[0]
        y[idx]=( c- x[idx])/float(c - b)
    idx=np.nonzero(x==b)
    y[idx]=1
    return y   




def trapez(x,rot,abc):
    y=np.zeros(len(x))
    if (rot=='ORTA'):
        assert len(abc)==3,'başlangıç bitiş ve tepe değerleri verilmelidir'
        a,b,c=np.r_[abc]
        assert a<=b and b<=c ,'üyelik değerleri küçükten büyüğe doğru a < b < c şeklinde olmalıdır'
        idx=np.nonzero(np.logical_and(x>=0,x < a))[0]
        y[idx]=(x[idx])/float(a)
        idx=np.nonzero(np.logical_and(x>=a,x < b))[0]
        y[idx]=1
        idx=np.nonzero(np.logical_and(x>=b,x < c))[0]
        y[idx]=(c-x[idx])/float(c-b)
        return y
    else:
        assert len(abc)==2,'başlangıç bitiş ve tepe değerleri verilmelidir'
        a,b=np.r_[abc]
        if(rot=="SOL"):
            assert a<=b,'a<b'
            idx=np.nonzero(x<a)[0]
            y[idx]=1
            idx=np.nonzero(np.logical_and(x>=a,x < b))[0]
            y[idx]=(x[idx]-b)/float(a-b)
            return y
        elif(rot=='SAĞ'):
            assert a<=b,'a<b'
            idx=np.nonzero(x>a)[0]
            y[idx]=1
            idx=np.nonzero(np.logical_and(x>=a,x <= b))[0]
            y[idx]=(x[idx]-a)/float(b-a)
            return y
        
def yamuk(x,abc):
    a,b,c,d=np.r_[abc]
    y=np.zeros(len(x))
    
    idx=np.nonzero(x<a)[0]
    y[idx]=0
    idx=np.nonzero(np.logical_and(x>=a,x<b))[0]
    y[idx]=(x[idx]-a)/float(b-a)
    idx=np.nonzero(np.logical_and(x>=b,x<=c))[0]
    y[idx]=1
    idx=np.nonzero(np.logical_and(c < x,x < d))[0]
    y[idx]=( d- x[idx])/float(d - c)
    idx=np.nonzero(x==b)
    y[idx]=1
    return y   

def uyelik(x,xmf,xx,zero_outside_x=True):
    if not zero_outside_x:
        kwargs=(None,None)
    else:
        kwargs=(0.0,0.0)
    return np.interp(xx,x,xmf,left=kwargs[0],right=kwargs[1])

def durulaştırma(x,LFX,model):
    model=model.lower()
    x=x.ravel()
    LFX=LFX.ravel()
    n=len(x)
    if n !=len(LFX):
        print("bulanık küme üyeliği ve değerleri eşit olmalıdır")
        return
# #'ağırlık merkezi':ağırlık merkezi ortalama
#  *'maxort':maksimum ortalama
#  *'minom':en büyüklerin en küçüğü
#  *'maxom':en küçüklerin en büyüğü
    if 'agirlik_merkezi' in model:
        if 'agirlik_merkezi' in model:
            return agirlik_merkezi(x,LFX)
        
        elif 'AC0' in model:
            return 0 #AC0(x,mfx)
        
    elif 'maxort' in model:
        return np.mean(x[LFX==LFX.max()])
    elif 'minom' in model:
        return np.min(x[LFX==LFX.max()])
    elif 'maxom' in model:
        return np.max(x[LFX==LFX.max()])
    
#Ağırlık merkezi durulaştırma metodu
def agirlik_merkezi(x,LFX):
    sum_moment_area=0.0
    sum_area=0.0
#X dizisinin 1 elemanlı olduğu durum,tek bir bulanık kümeye ait olduğu durumdur.
#Eğer üyelik fonksiypnu tek bir bulanık kümeye ait ise de 
#ağırlık merkezi hesabı yapmaya gerek yoktur.Kendi ağırlık alanı hesaplanıp döndürülür.
    if len(x)==1:
        return x[0]*LFX[0]/np.fmax(LFX[0], np.finfo(float).eps).astype(float)
#birden fazla bulanık küme var ise;
#ilgilik üyelik değerinin çıkış kümesi üzerinden kestiği alanlar hesaplanır.
    for i in range(1,len(x)):
        x1=x[i-1]
        x2=x[i]
        y1=LFX[i-1]
        y2=LFX[i]

        if not(y1 == y2 == 0.0 or x1 == x2):
            if y1 == y2:  # Dikdörtgen alan ise :
                moment = 0.5 * (x1 + x2)
                area = (x2 - x1) * y1
            elif y1 == 0.0 and y2 != 0.0:  # Üçgen, yükseklik y2 ise :
                moment = 2.0 / 3.0 * (x2-x1) + x1
                area = 0.5 * (x2 - x1) * y2
            elif y2 == 0.0 and y1 != 0.0:  # Üçgen, yükseklik y1 ise :
                moment = 1.0 / 3.0 * (x2 - x1) + x1
                area = 0.5 * (x2 - x1) * y1
            else:                          # Diğer Koşullarda
                moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                area = 0.5 * (x2 - x1) * (y1 + y2)
            #toplam alan+=hesaplanan kesme alanı
            sum_moment_area+=moment*area
            sum_area+=area
    return sum_moment_area / np.fmax(sum_area,
                                     np.finfo(float).eps).astype(float)
                

    



















        