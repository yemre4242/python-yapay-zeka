import numpy as np
import fuzzPY as fuzz
import matplotlib.pylab as plt

x_Alan=np.arange(0,781,1)
x_Konum=np.arange(0,1001,1)
x_şekil=np.arange(0,741,1)
x_KAKS=np.arange(0.0,2.0,0.01)
x_O=np.arange(0,10,0.1)
 
# ALAN
x_Alan_küçük=fuzz.trapez(x_Alan,"SOL",[200,300])
x_Alan_orta=fuzz.yamuk(x_Alan,[200,300,400,500])
x_Alan_büyük=fuzz.trapez(x_Alan,"SAĞ",[400,500])

#Konum özellikleri
x_konum_merkezi_değil=fuzz.trapez(x_Konum,"SOL",[200,300])
x_konum_az_merkezi=fuzz.yamuk(x_Konum,[200,300,400,500])
x_konum_merkezi=fuzz.trapez(x_Konum,"SAĞ",[400,500])
 
#Şekil
x_şekil_düzgün_değil=fuzz.trapez(x_şekil,"SOL",[250,300])
x_şekil_az_düzgün=fuzz.yamuk(x_şekil,[250,300,400,500])
x_şekil_düzgün=fuzz.trapez(x_şekil,"SAĞ",[400,500])

#KAKS
x_KAKS_az=fuzz.trapez(x_KAKS,"SOL",[0.5,0.7])
x_KAKS_orta=fuzz.yamuk(x_KAKS,[0.5,0.7,0.8,1.0])
x_KAKS_çok=fuzz.trapez(x_KAKS,"SAĞ",[0.8,1.0])

#çıkış
O_az=fuzz.trapez(x_O,"SOL",[2.5,5.0])
O_orta=fuzz.uçgen(x_O,[2.5,5.0,8.5])
O_çok=fuzz.trapez(x_O,"SAĞ",[5.0,8.5])

fig,(ax0,ax1,ax2,ax3,ax4)=plt.subplots(nrows=5,figsize=(6,10))

ax0.plot(x_Alan,x_Alan_küçük,"g",linewidth=2,label="küçük")
ax0.plot(x_Alan,x_Alan_orta,"b",linewidth=2,label="orta")
ax0.plot(x_Alan,x_Alan_büyük,"r",linewidth=2,label="büyük")
ax0.set_title("ALAN")
ax0.legend()

ax1.plot(x_Konum,x_konum_merkezi_değil,"g",linewidth=2,label="merkezi değil")
ax1.plot(x_Konum,x_konum_az_merkezi,"b",linewidth=2,label="az merkezi")
ax1.plot(x_Konum,x_konum_merkezi,"r",linewidth=2,label="merkezi")
ax1.set_title("KONUM")
ax1.legend()

ax2.plot(x_şekil,x_şekil_düzgün_değil,"g",linewidth=2,label="düzgün değil")
ax2.plot(x_şekil,x_şekil_az_düzgün,"b",linewidth=2,label="az düzgün")
ax2.plot(x_şekil,x_şekil_düzgün,"r",linewidth=2,label="düzgün")
ax2.set_title("ŞEKİL")
ax2.legend()

ax3.plot(x_KAKS,x_KAKS_az,"g",linewidth=2,label="0.0 - 0.5")
ax3.plot(x_KAKS,x_KAKS_orta,"b",linewidth=2,label="0.5 - 1.0")
ax3.plot(x_KAKS,x_KAKS_çok,"r",linewidth=2,label="0.1 - 2.0")
ax3.set_title("KAKS")
ax3.legend()

ax4.plot(x_O,O_az,'r',linewidth=2,label="az")
ax4.plot(x_O,O_orta,'g',linewidth=2,label="orta")
ax4.plot(x_O,O_çok,'b',linewidth=2,label="çok")
ax4.set_title("Çıkış hız sınırı")
ax4.legend()

plt.tight_layout()
plt.savefig("arsa değerleme üyelik fonksiyonları.png")

input_alan=input("alanı giriniz:(0-780)")
input_konum=input("Konumunu giriniz(0-1000):")
input_şekil=input("şeklinin düzgünlüğünü giriniz(0-740):")
input_KAKS=input("KAKS değerini giriniz(0.0-2.0):")


#Alan
fit_x_alan_küçük=fuzz.uyelik(x_Alan,x_Alan_küçük,input_alan)
fit_x_alan_orta=fuzz.uyelik(x_Alan,x_Alan_orta,input_alan)
fit_x_alan_büyük=fuzz.uyelik(x_Alan,x_Alan_büyük,input_alan)

#Konum
fit_x_konum_merkezi_değil=fuzz.uyelik(x_Konum,x_konum_merkezi_değil,input_konum)
fit_x_konum_az_merkezi=fuzz.uyelik(x_Konum,x_konum_az_merkezi,input_konum)
fit_x_konum_merkezi=fuzz.uyelik(x_Konum,x_konum_merkezi,input_konum)

#şekil
fit_x_şekil_düzgün_değil=fuzz.uyelik(x_şekil,x_şekil_düzgün_değil,input_şekil)
fit_x_şekil_az_düzgün=fuzz.uyelik(x_şekil,x_şekil_az_düzgün,input_şekil)
fit_x_şekil_düzgün=fuzz.uyelik(x_şekil,x_şekil_düzgün,input_şekil)

#KAKS
fit_x_kaks_az=fuzz.uyelik(x_KAKS,x_KAKS_az,input_KAKS)
fit_x_kaks_orta=fuzz.uyelik(x_KAKS,x_KAKS_orta,input_KAKS)
fit_x_kaks_çok=fuzz.uyelik(x_KAKS,x_KAKS_çok,input_KAKS)

kural1=np.fmin(np.fmin(fit_x_alan_küçük,fit_x_konum_merkezi_değil),O_az)
kural2=np.fmin(np.fmin(fit_x_alan_orta,fit_x_konum_az_merkezi),O_orta)
kural3=np.fmin(np.fmin(fit_x_alan_büyük,fit_x_konum_merkezi),O_çok)
kural4=np.fmin(np.fmin(fit_x_şekil_düzgün_değil,fit_x_kaks_az),O_az)
kural5=np.fmin(np.fmin(fit_x_şekil_az_düzgün,fit_x_kaks_orta),O_orta)
kural6=np.fmin(np.fmin(fit_x_şekil_düzgün,fit_x_kaks_çok),O_çok)

out_az=np.fmax(kural1,kural4)
out_orta=np.fmax(kural2,kural5)
out_çok=np.fmax(kural3,kural6)



O_zeros=np.zeros_like(x_O)
fig,grafik_output=plt.subplots(figsize=(7,4))
grafik_output.fill_between(x_O,O_zeros,out_az,facecolor='r',alpha=0.7)
grafik_output.plot(x_O,O_az,'r',linestyle="--")
grafik_output.fill_between(x_O,O_zeros,out_orta,facecolor='g',alpha=0.7)
grafik_output.plot(x_O,O_orta,'g',linestyle="--")
grafik_output.fill_between(x_O,O_zeros,out_çok,facecolor='b',alpha=0.7)
grafik_output.plot(x_O,O_çok,'b',linestyle="--")
plt.tight_layout()
plt.savefig("C:/Users/yuemr/OneDrive - Yildiz Technical University/Masaüstü/yapay_zeka/arsa/çıkış arsa.png")


mutlak_bulanık_sonuc=np.fmax(out_az,out_orta,out_çok)
durulaştırılmış_sonuc=fuzz.durulaştırma(x_O,mutlak_bulanık_sonuc,'agirlik_merkezi')
durulaştırılmış_sonuc=durulaştırılmış_sonuc*3/2

sonuc_az=fuzz.uyelik(x_O,O_az,durulaştırılmış_sonuc)
sonuc_orta=fuzz.uyelik(x_O,O_orta,durulaştırılmış_sonuc)
sonuc_çok=fuzz.uyelik(x_O,O_çok,durulaştırılmış_sonuc)


hız_sınırı=1000000
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
print("mevcut şartlar altında arsa değeri,",degisim,"değiştirilerek",hız_sınırı,"olmalıdır")
print("değişim oranı:%",float(degisim/100))

