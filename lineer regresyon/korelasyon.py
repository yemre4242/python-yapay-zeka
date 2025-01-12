import random
def ortalama_hesabı(dizi):
    elemansayisi=len(dizi)
    if elemansayisi<1:
        return 0
    else:
        return sum(dizi)/elemansayisi
    
def standart_sapma_hesabi(dizi):
    sapma=0.0
    eleman_Sayisi=len(dizi)
    if eleman_Sayisi<=1:
        return 0
    else:
        for eleman in dizi:
            sapma = float(eleman)-ortalama_hesabı(dizi)**2
        sapma=(sapma/float(eleman))**0.5
        return sapma
    
def korelasyon_bul(dizi1,dizi2):
    assert len(dizi1)==len(dizi2)
    elemansayisi=len(dizi1)
    assert elemansayisi>0
    dizi_olasilik=0
    dizi1_dağilim=0
    dizi2_dağilim=0
    for i in range(elemansayisi):
        dizi1_fark=dizi1[i]-ortalama_hesabı(dizi1)
        dizi2_fark=dizi2[i]-ortalama_hesabı(dizi2)
        dizi_olasilik+=dizi1_fark*dizi2_fark
        dizi1_dağilim+=dizi1_fark*dizi1_fark
        dizi2_dağilim+=dizi2_fark*dizi2_fark

    return dizi_olasilik/(dizi1_dağilim*dizi2_dağilim)**.5
    
dizi1=sorted([random.randrange(1,500,1) for i in range(500)])
dizi2=sorted([random.randrange(1,500,1) for i in range(500)])


print("korelasyon:")
print(korelasyon_bul(dizi1,dizi2))
print("standart sapma -1:")
print(standart_sapma_hesabi(dizi1))
print("standart sapma -2:")
print(standart_sapma_hesabi(dizi2))