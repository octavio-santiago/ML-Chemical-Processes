#Python 3.7
#Octavio Bomfim Santiago
#12/01/2018 - Trabalho Modelagem Fluxo de Massa FCC multicomponente
import sympy
from sympy import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import csv


# Defininfo algumas funções matemáticas
pi = math.pi
def exp(x):
    e = math.exp(x)
    return e

#Definindo as variáveis físicas constantes p,D,v,u,vi,dab, ya,yb,yc,yd,Dya, C, Oc, Pr, k , T0
ya=0.108663
yb=0.474592
yc=0.259723
yd=0.094574
Dya=0.364811

R = 8.314
#Massa molecular kg/kmol
Ma=371
Mb=106.7
Mc=178.6
Md=14.4
Mm = 1/((ya/Ma)+(yb/Mb)+(yc/Mc)+(yd/Md))
#dab,dac,dad
dab=1
dac=1
dad=1
daList = [dab,dac,dad]
#densidade do catalisador kg/m3
pc=720
#F = kg/s
Fg = 62.5
Fc = 400.32
CO = Fc/Fg
#velocidade = v = m/s
v=24.95
#D = m diametro do catalisador
D = 0.000065
#p = kg/m3 densidade do gas
p = 0.9019

#Concentração kmol/m3
c = p/Mm
#u = m2/s
u = 3.44 *(pow(10,(-5)))

vi=u/p
#diametro do riser (m)
Dr = 1.36

A = pi*((Dr)**2)/4
#print('A: ',A)
#Temperaturas = Kelvin
Tinf = 748.2

#Calcula o valor de Reynolds
def ReCalc (p,D,v,u,vi):
    if (vi == 0 ) :
        Re = (p*D*v)/u
        return Re
    else:
        Re = (v*D)/vi
        return Re

#Calcula o valor de Prandtl (gases - Pr ranges 0.7 - 1.0)
def PrCalc (Tinf,k):
    #Capacidade termica J/kg K do gas
    Tvap=0.2*(368+453+472+528+644)
    Sl=0.0125*(644-368)
    Tmeabp = Tvap - 0.5556*exp(-0.9440-0.0087*((1.8*Tvap - 491.67)**0.6667)+ 2.9972*(Sl**0.3333))
    Kf=((1.8*Tmeabp)**(1/3))/p
    Kf = 3.5
    B4 =(((12.8/Kf)-1)*(1-(10/Kf))*(p-0.0885)*(p-0.7)*(10**4))**2
    B4=0
    B3 = (1.356523*(10**(-6)))*(1.6946 + 0.0884*B4)
    B2 = (-7.53624*(10**(-4)))*(2.9247-(1.5524-0.05543*Kf)*Kf + B4*(6.0283-(5.0694/p)))
    B1= -1.492343 + 0.124432 *11 + B4*(1.23519 - (1.04025/p))
    cp = (B1 + B2*Tinf + B3*(Tinf**2))*1000
    #print('vi: ',vi)
    #print('Cp: ',cp)
    #cp = 1150
    #cp=5000
    Pr = (vi* cp)/k
    #μ = absolute or dynamic viscosity (kg/m s)
    #cp = specific heat (J/kg K)
    #k = thermal conductivity (W/m K)
    
    return Pr
#Calcula os daxs
def daListCalc(Tinf):
    daList=[]
    T=Tinf
    Va= 230 * (10**(-3))
    Vb= 140.06 * (10**(-3))
    Vc= 156 * (10**(-3))
    #Gilliand
    dab= (4.3*(10**(-9)))*((T**(3/2))/((((Va**(1/3))+(Vb**(1/3))))**2))*(((1/Ma)+(1/Mb))**(1/2))
    dac= (4.3*(10**(-9)))*((T**(3/2))/((((Va**(1/3))+(Vc**(1/3))))**2))*(((1/Ma)+(1/Mc))**(1/2))
    dad=10**(-17)
    daList.append(dab)
    daList.append(dac)
    daList.append(dad)
    return daList

#Calcula o valor de Sc
def ScCalc (vi,dab):
    Sc = vi/dab
    #print('Sc: ',Sc)
    return Sc

#Calcula os hms
def hmCalc (daList):
    daList = daListCalc(Tinf)
    hmList = []
    for dax in daList:
        Re = ReCalc(p,D,v,u,vi)
        Sc = ScCalc(vi,dax)
        hm = (2 + (0.552 * (pow(Re,(1/2))) * (pow(Sc,(1/3)))) * (dax/D))
        hmList.append(hm)
    
    return hmList

#Desativação do Catalisador por coque
def OcCalc():
    alfac0 = 1.1 * (exp(-5))
    alfac = 0.1177
    Cck0 = 0.001
    Cck = Cck0 + (Fg*yd/Fc)
    Oc = exp(-alfac * Cck)
    return Oc
    

#Variação de Temperatura
#Calculo de h
def hCalc (Tinf):
    Re = ReCalc(p,D,v,u,vi)
    #print('Re: ', Re)
    #print('Mm: ', Mm)
    k = (10**(-3))*((1.9469-(0.374*Mm)) + (1.4815*(10**(-3))*(Mm**2))+(0.1028*Tinf))
    #print('k: ', k)
    Pr = PrCalc (Tinf,k)
    #print('Pr: ' , Pr)
    h = (k/Dr) *(2+(0.4*(Re**(0.5)))+ 0.06*(Re**(2/3)))*(Pr**(0.4))*1
    #print('h: ' , h)
    return h

def KrCalc(Tinf,Oc):
    T = Tinf
    #Parametros da literatura (Du et al.)
    # s**-1
    k01=7957.29
    k02=14433.4
    k03=40.253
    
    k04=197.933
    k05=75.282
    k06=2.031

    #kJ/kmol
    E1=53927.7
    E2=57186.6
    E3=35433.6
    E4=48144.5
    E5=61159.4
    E6=61785.1

    # kJ/kmol
    dH1 = 190.709
    dH2 = 128.45
    dH3 = 458.345
    dH4 = 513.568
    dH5 = 305.925
    dH6 = 117.212
    
    #Gasoleo
    k1 = k01 * exp(-E1/(R*T))
    k2 = k02 * exp(-E2/(R*T))
    k3 = k03 * exp(-E3/(R*T))
    #Diesel
    k4 = k04 * exp(-E4/(R*T))
    k5 = k05 * exp(-E5/(R*T))
    #Gasolina
    k6 = k06 * exp(-E6/(R*T))

    Kr = (dH1*k1*(ya**2) + dH2*k2*(ya**2) + dH3 * k3 * (ya**2) + dH4*k4*yc + dH5*k5*yc + dH6 * k6 * yb)*Oc
    #print("Kr ",Kr)
    kList = [k1,k2,k3,k4,k5,k6]
    #print("K: ",kList)
    return Kr

def MSL(Tinf,ya,yb,yc,yd):
    h = hCalc(Tinf)
    Oc = OcCalc()
    Kr = KrCalc(Tinf,Oc)
    #Calcula os hms
    daList = daListCalc(Tinf)
    hmList = hmCalc(daList)
    #print('daList: ',daList)
    hmab = hmList[0]
    hmac = hmList[1]
    hmad = hmList[2]

    #montando o sistema linear
    a11 = ((yb/(c*hmab))+(yc/(c*hmac))+(yd/(c*hmad)))
    a12 = (-ya/(c*hmab))
    a13 = (-ya/(c*hmac))
    a14 = (-ya/(c*hmad))
    a15 = 0
    a21 = 1
    a22 = 4.1
    a23 = 2.2
    a24 = -1
    a25 = 0
    a31 = 0
    a32 = 1.9
    a33 = 1
    a34 = -1
    a35 = 0
    a41 = 0
    #a42 = 1
    a42=0
    a43 = 0
    #a44 = -1
    a44 = 1
    a45 = 0
    a51 = 0
    a52 = 0
    a53 = 0
    a54 = 0
    a55 = 1
    
    
    y1 = -Dya
    y2 = 0.0
    y3 = 0.0
    y4 = 0.0
    y5 = Tinf + (Kr/h)
  
    #resolvendo o sistema linear
    a = np.array([[a11,a12,a13,a14,a15],
                 [a21,a22,a23,a24,a25],
                 [a31,a32,a33,a34,a35],
                 [a41,a42,a43,a44,a45],
                 [a51,a52,a53,a54,a55]],dtype=float)
    b = np.array([y1,y2,y3,y4,y5],dtype=float)


    x = np.linalg.solve(a, b)
    print(" ")
    print( 'Na (Gasoleo): ' , x[0] ,' kgA/m2 s')
    print( 'Nb (Gasolina): ' , x[1], ' kgB/m2 s')
    print( 'Nc (Diesel): ' , x[2], ' kgC/m2 s')
    print( 'Nd (Coque): ' , x[3], ' kgD/m2 s')
    print( 'Ts: ' , x[4])
    print(" ")
    #print('A ', A, ' m2')
    ma= x[0] * A
    mb = x[1] * A
    mc = x[2] * A
    md = x[3] * A
    print( 'ma (Gasoleo): ' , ma ,' kgA/m2 s')
    print( 'mb (Gasolina): ' , mb, ' kgB/m2 s')
    print( 'mc (Diesel): ' , mc, ' kgC/m2 s')
    print( 'md (Coque): ' , md, ' kgD/m2 s')
    print(" ")
    return x
                          
    

#variando composições (talvez)
#varinado Oc que varia com as composicoes
#variando C/O

##a = np.array([[3,1], [1,2]])
##b = np.array([9,8])
##x = np.linalg.solve(a, b)
##print(x[0])

MSL(Tinf,ya,yb,yc,yd)

#variando Tinf
Ty = []
Tx =[]
Nay=[]
Nby=[]
Ncy=[]
for T in range(300,1000,100):
    answer = MSL (T,ya,yb,yc,yd)
    Tn = answer[4]
    Tx.append(T)
    
    Ty.append(Tn)
    Nra = answer[0]
    Nay.append(Nra)
    Nrb = answer[1]
    Nby.append(Nrb)
    Nrc = answer[2]
    Ncy.append(Nrc)


#variando altura do Riser que varia as composições 
Compy = [[0.6,0.25,0.13,0.02],
         [0.178970657,0.308802981,0.377852818,0.087098277],
         [0.113064741,0.304261761,0.42920354,0.092221705],
         [0.08430368,0.289590126,0.463670238,0.095132743],
         [0.068351188,0.273521192,0.49021891,0.09711225],
         [0.058104332,0.258150908,0.512109921,0.09862599],
         [0.051001397,0.24394504,0.530507685,0.099790405],
         [0.045761528,0.230903586,0.546576619,0.100838379],
         [0.041686074,0.219026549,0.560666046,0.10165347],
         [0.040288775,0.21460177,0.565905915,0.102002795]
         ]
Compx = [5,10,15,20,25,30,35,40,45,47]
Ty2 = []
Nay2=[]
Nby2=[]
Ncy2=[]
for hRa in Compy:
    ya=hRa[0]
    yb=hRa[1]
    yc=hRa[2]
    yd=hRa[3]
    answer = MSL (Tinf,ya,yb,yc,yd)
    Tn = answer[4]
    Ty2.append(Tn)
    
    Nra = answer[0]
    Nay2.append(Nra)
    Nrb = answer[1]
    Nby2.append(Nrb)
    Nrc = answer[2]
    Ncy2.append(Nrc)



#carregando dados da simulação
dataframe = pd.DataFrame()
dataframe['x'] = Tx
dataframe['y'] = Ty
x_values = dataframe[['x']]
y_values = dataframe[['y']]

#treinando o modelo
model = linear_model.LinearRegression()
model.fit(x_values, y_values)
a = model.coef_[0][0]
b = model.intercept_[0]
modelY=[]
for x in Tx:
    y = a*x+b
    modelY.append(y)
#temperaturas
plt.figure(1)
plt.subplot(221)
plt.title('Temperatura de Superfície')
plt.xlabel('Temperatura Inf')
plt.ylabel('Temperatura Superfície (K)')
plt.grid(True)
plt.plot(Tx,Ty,'bo',label="Ts")
plt.plot(Tx,modelY,'k',label="reg lin")
plt.legend()

plt.subplot(222)
plt.title('Fluxo total de Massa')
plt.xlabel('Temperatura Inf')
plt.ylabel('Nx Fluxo - kgX/m2s')
plt.grid(True)
plt.plot(Tx,Nay,'ro',label='Gasoleo')
plt.plot(Tx,Nby,'bs',label='Gasolina')
plt.plot(Tx,Ncy,'g^',label='Diesel')
plt.legend()
#plt.legend((Nay,Nby,Ncy),('Gasoleo','Gasolina','Diesel'))
#plt.plot(Tx,Ncy,'ro')

plt.subplot(223)
plt.title('Fluxo total de Massa')
plt.xlabel('Altura Riser (m)')
plt.ylabel('Nx Fluxo - kgX/m2s')
plt.grid(True)
plt.plot(Compx,Nay2,'ro',label='Gasoleo')
plt.plot(Compx,Nby2,'bs',label='Gasolina')
plt.plot(Compx,Ncy2,'g^',label='Diesel')
plt.legend()
#plt.plot(Tx,Ncy,'ro')

plt.subplot(224)
plt.title('Temperatura Catalisador')
plt.xlabel('Altura Riser (m)')
plt.ylabel('Temperatura Catalisador (K)')
plt.grid(True)
plt.subplots_adjust(hspace=0.82,wspace=0.82)
plt.plot(Compx,Ty2,'ro',label='Ts')
plt.legend()
#plt.plot(Tx,Ncy,'ro')


plt.show()

print(dataframe.shape[1])

def SalvarCsv():
    with open('fcc.csv', mode='w') as fcc_file:
        fcc_writer = csv.writer(fcc_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fcc_writer.writerow(['Tx', 'Ty'])
        for x in range(0,len(dataframe.index)):
            fcc_writer.writerow(dataframe.iloc[x])


    with open('fcc2.csv', mode='w') as fcc2_file:
        fcc2_writer = csv.writer(fcc2_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fcc2_writer.writerow(['ya0','yb0','yc0','yd0', 'h','Tx', 'Ty','Nay','Nby','Ncy'])
        for x in range(0,len(Compy)):
            listaX = []
            listaX = [Compy[x][0],Compy[x][1],Compy[x][2],Compy[x][3],Compx[x],Tinf,Ty2[x],Nay2[x],Nby2[x],Ncy2[x]]
            print(listaX)
            fcc2_writer.writerow(listaX)






    
