import numpy as np
import fuzzPY as fuzz
import matplotlib.pyplot as plt
x_R=np.arange(0,91,1)
x_W=np.arange(0,11,1)
x_S=np.arange(0,151,1)
x_E=np.arange(0,21,1)
x_O=np.arange(0,101,1)

R_kotu=fuzz.trapez(x_R,"SOL",[30,45])
R_normal=fuzz.uçgen(x_R,[30,45,60])
R_iyi=fuzz.trapez(x_R,"SAĞ",[45,60])

#hava durumu üyelik fonksiyonu
W_kotü=fuzz.uçgen(x_W,[0,0,5])
w_normal=fuzz.uçgen(x_W,[0,5,10])
w_iyi=fuzz.uçgen(x_W,[5,10,10])

#ortalama hız üyelik fonksiyonu
S_az=fuzz.uçgen(x_S,[0,0,70])
S_orta=fuzz.uçgen(x_S,[0,70,130])
S_çok=fuzz.trapez(x_S,"SAĞ",[70,130])

#Kullanıcı tecrube uyelik fonksiyonu
E_az=fuzz.uçgen(x_E,[0,0,10])
E_orta=fuzz.uçgen(x_E,[0,10,20])
E_çok=fuzz.uçgen(x_E,[10,20,20])

#çıkı kümesi 
O_az=fuzz.trapez(x_O,"SOL",[25,50])
O_orta=fuzz.uçgen(x_O,[25,50,85])
O_çok=fuzz.trapez(x_O,"SAĞ",[50,85])

#yol eğim ve viraj
fig,(ax0,ax1,ax2,ax3,ax4)=plt.subplots(nrows=5,figsize=(6,10))
ax0.plot(x_R,R_kotu,'r',linewidth=2,label="kötü")
ax0.plot(x_R,R_normal,'g',linewidth=2,label="normal")
ax0.plot(x_R,R_iyi,'b',linewidth=2,label="iyi")
ax0.set_title("yol viraj ve eğimi")
ax0.legend()


#Hava şartları
ax1.plot(x_W,W_kotü,'r',linewidth=2,label="kötü")
ax1.plot(x_W,w_normal,'g',linewidth=2,label="normal")
ax1.plot(x_W,w_iyi,'b',linewidth=2,label="iyi")
ax1.set_title("Hava şartları")
ax1.legend()


#Ortalama hız grafiği
ax2.plot(x_S,S_az,'r',linewidth=2,label="az")
ax2.plot(x_S,S_orta,'g',linewidth=2,label="orta")
ax2.plot(x_S,S_çok,'b',linewidth=2,label="çok")
ax2.set_title("Sürücü ortalama hızı")
ax2.legend()


#Kullanıcı tecrubesi grafiği
ax3.plot(x_E,E_az,'r',linewidth=2,label="az")
ax3.plot(x_E,E_orta,'g',linewidth=2,label="orta")
ax3.plot(x_E,E_çok,'b',linewidth=2,label="çok")
ax3.set_title("Kullanıcı tecrubesi")
ax3.legend()

#çıkış hız sınırı grafiği
ax4.plot(x_O,O_az,'r',linewidth=2,label="az")
ax4.plot(x_O,O_orta,'g',linewidth=2,label="orta")
ax4.plot(x_O,O_çok,'b',linewidth=2,label="çok")
ax4.set_title("Çıkış hız sınırı")
ax4.legend()

#inputları al
input_R=input("Yol viraj düzeyini girin(0-90):")
input_W=input("Hava durumunu girin(0-10):")
input_S=input("Sürücü ortalama hızını girin(30-150):")
input_E=input("Kullanıcı deneyim yılını girin(0-20):")

R_fit_kötü=fuzz.uyelik(x_R,R_kotu,input_R)
R_fit_normal=fuzz.uyelik(x_R,R_normal,input_R)
R_fit_iyi=fuzz.uyelik(x_R,R_iyi,input_R)

#Hava durumu
W_fit_kötü=fuzz.uyelik(x_W,W_kotü,input_W)
W_fit_normal=fuzz.uyelik(x_W,w_normal,input_W)
W_fit_iyi=fuzz.uyelik(x_W,w_iyi,input_W)

#ortalama hız
S_fit_az=fuzz.uyelik(x_S,S_az,input_S)
S_fit_ortalama=fuzz.uyelik(x_S,S_orta,input_S)
S_fit_çok=fuzz.uyelik(x_S,S_çok,input_S)

#sürücü deneyimi
E_fit_az=fuzz.uyelik(x_E,E_az,input_E)
E_fit_ortalama=fuzz.uyelik(x_E,E_orta,input_E)
E_fit_çok=fuzz.uyelik(x_E,E_çok,input_E)


rule1=np.fmin(np.fmin(R_fit_kötü,W_fit_kötü),O_az)
rule2=np.fmin(np.fmin(R_fit_normal,W_fit_normal),O_orta)
rule3=np.fmin(np.fmin(R_fit_iyi,W_fit_iyi),O_çok)
rule4=np.fmin(np.fmax(S_fit_az,E_fit_az),O_az)
rule5=np.fmin(np.fmax(S_fit_ortalama,E_fit_ortalama),O_orta)
rule6=np.fmin(np.fmax(S_fit_çok,E_fit_çok),O_çok)


out_az=np.fmax(rule1,rule4)
out_ortalama=np.fmax(rule2,rule5)
out_çok=np.fmax(rule3,rule6)

O_ZEROS=np.zeros_like(x_O)
fig, grafik_output=plt.subplots(figsize=(7,4))
grafik_output.fill_between(x_O,O_ZEROS,out_az,facecolor='r',alpha=0.7)
grafik_output.plot(x_O,O_az,'r',linestyle='--')
grafik_output.fill_between(x_O,O_ZEROS,out_ortalama,facecolor='g',alpha=0.7)
grafik_output.plot(x_O,O_orta,'g',linestyle='--')
grafik_output.fill_between(x_O,O_ZEROS,out_çok,facecolor='b',alpha=0.7)
grafik_output.plot(x_O,O_çok,'b',linestyle='--')
plt.tight_layout()
plt.savefig("çıkış2.png")

mutlak_bulanık_sonuc=np.fmax(out_az,out_ortalama,out_çok)

durulaştırılmış_sonuc=fuzz.durulaştırma(x_O,mutlak_bulanık_sonuc,'agirlik_merkezi')
durulaştırılmış_sonuc=durulaştırılmış_sonuc*3/2

sonuc_az=fuzz.uyelik(x_O,O_az,durulaştırılmış_sonuc)
sonuc_orta=fuzz.uyelik(x_O,O_orta,durulaştırılmış_sonuc)
sonuc_çok=fuzz.uyelik(x_O,O_çok,durulaştırılmış_sonuc)

hız_sınırı=100
hız_sınırı_düşük=hız_sınırı-(sonuc_az*durulaştırılmış_sonuc)
hız_sınırı_yüksek=hız_sınırı+(sonuc_çok*durulaştırılmış_sonuc)
hız_sınırı = (hız_sınırı_düşük+ hız_sınırı_yüksek) /2

if (sonuc_az> sonuc_çok):
    hız_sınırı=hız_sınırı+(sonuc_orta*durulaştırılmış_sonuc) / 3
else:
    hız_sınırı=hız_sınırı-(sonuc_orta*durulaştırılmış_sonuc) / 3

degisim=hız_sınırı-100

print("-"*50)
print("Duru sonuç üyelik değeri-->az:",sonuc_az,"\nDuru sonuç üyelik değeri-->orta:",sonuc_orta,"\nDuru sonuç üyelik değeri-->çok:",sonuc_çok)


print("-"*50)

print("Duru sonuç->",durulaştırılmış_sonuc)

print("-"*50)
print("mevcut şartlar altında hız sınırı,",degisim,"değiştirilerek",hız_sınırı,"olmalıdır")
print("değişim oranı:%",float(degisim/100))





