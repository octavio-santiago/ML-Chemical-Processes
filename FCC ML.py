import sympy
from sympy import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import csv

fcc_df = pd.read_csv("fcc2.csv")

#plotar os graficos
def plotarFig (df2):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    df2.plot(kind='scatter',x='index',y='Ty',color='red',ax=axes[0,0])
    df2.plot(kind='line',x='index',y='Ty',color='blue',ax=axes [0,0])
    df2.plot(kind='line',x='index',y='Ty0',color='green',ax=axes[0,0])
    
    df2.plot(kind='line',x='index',y='Ncy0',color='red',ax=axes[0,1])
    df2.plot(kind='line',x='index',y='Ncy',color='blue',ax=axes[0,1])
    plt.show()

#linear regression
from sklearn.linear_model import LinearRegression
def LinReg(df,z,Ncy0,Ty0):
    model = LinearRegression()
    model2 = LinearRegression()
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    model.fit(X,y)
    model2.fit(X,y2)
    predicted2 = model2.predict(z)
    predicted = model.predict(z)
    print('-----------------------------')
    print("Regressão Linear")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))

    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2
    
    
    
print(" ")
print("Regressão Linear de uma composição existente")
z = pd.DataFrame(data = [[0.6,0.25,0.13,0.02,5,748.2]])
df2 = LinReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = LinReg(fcc_df,z,0.001,1109)



##print("Regressão Linear variando a composição")
##z = [[0.6,0.2,0.17,0.03,5,748.2]]
##LinReg(fcc_df,z)
##
##print("Regressão Linear variando a composição e altura")
##z = [[0.6,0.2,0.17,0.03,20,748.2]]
##LinReg(fcc_df,z)
##
##print("Regressão Linear variando a composição e temperatura")
##z = [[0.6,0.2,0.17,0.03,5,500]]
##LinReg(fcc_df,z)


#Lasso
from sklearn import linear_model
def LassoReg(df,z,Ncy0,Ty0):
    reg = linear_model.Lasso(alpha=0.1)
    reg2 = linear_model.Lasso(alpha=0.1)
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    reg.fit(X,y)
    reg2.fit(X,y2)
    predicted = reg.predict(z)
    predicted2 = reg2.predict(z)
    print('-----------------------------')
    print("Regressão de Lasso")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2

print(" ")
print("Regressão de Lasso de uma composição existente")
df2 = LassoReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = LassoReg(fcc_df,z,0.001,1109)


#Lasso CV: coordinate descent
from sklearn import linear_model
def LassoCVReg(df,z,Ncy0,Ty0):
    reg = linear_model.LassoCV(cv=10)
    reg2 = linear_model.LassoCV(cv=10)
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    reg.fit(X,y)
    reg2.fit(X,y2)
    predicted = reg.predict(z)
    predicted2 = reg2.predict(z)
    print('-----------------------------')
    print("Regressão de Lasso CV")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2

print(" ")
print("Regressão de Lasso CV de uma composição existente")
df2 = LassoCVReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = LassoCVReg(fcc_df,z,0.001,1109)


#Lasso IC: least angle regression with BIC/AIC criterion
from sklearn import linear_model
def LassoICReg(df,z,Ncy0,Ty0):
    reg = linear_model.LassoLarsIC(criterion='bic')
    reg2 = linear_model.LassoLarsIC(criterion='bic')
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    reg.fit(X,y)
    reg2.fit(X,y2)
    predicted = reg.predict(z)
    predicted2 = reg2.predict(z)
    print('-----------------------------')
    print("Regressão de Lasso IC (BIC)")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2

print(" ")
print("Regressão de Lasso IC (BIC) de uma composição existente")
df2 = LassoICReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = LassoICReg(fcc_df,z,0.001,1109)



#usando LARS Lasso
from sklearn import linear_model
def LarsReg(df,z,Ncy0,Ty0):
    reg = linear_model.LassoLars(alpha=.1)
    reg2 = linear_model.LassoLars(alpha=.1)
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    reg.fit(X,y)
    reg2.fit(X,y2)
    predicted = reg.predict(z)
    predicted2 = reg2.predict(z)
    print('-----------------------------')
    print("Regressão de LARS")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2
##    erro = predicted - 1109.833738
##    erroPerc = (abs(erro)/1109.833738)*100
##    print( "Erro: ", erro ,  "Porc: " , erroPerc , " %")

print(" ")
print("Regressão de LARS - Lasso de uma composição existente")
df2 = LarsReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = LarsReg(fcc_df,z,0.001,1109)


#usando Ridge Regression CV
from sklearn import linear_model
def RidgeReg(df,z,Ncy0,Ty0):
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    reg2 = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    reg.fit(X,y)
    reg2.fit(X,y2)
    predicted = reg.predict(z)
    predicted2 = reg2.predict(z)
    print('-----------------------------')
    print("Regressão de Ridge")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2

print(" ")    
print("Regressão de Ridge de uma composição existente")
df2 = RidgeReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = RidgeReg(fcc_df,z,0.001,1109)



#usando SVM Regression
from sklearn import svm
def SvmReg(df,z,Ncy0,Ty0):
    clf = svm.SVR()
    clf2 = svm.SVR()
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    clf.fit(X,y)
    clf2.fit(X,y2)
    predicted = clf.predict(z)
    predicted2 = clf2.predict(z)
    print('-----------------------------')
    print("Regressão de Svm")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2


print(" ")    
print("Regressão de SVM de uma composição existente")
df2 = SvmReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = SvmReg(fcc_df,z,0.001,1109)

#usando Bayesian Ridge Regression
from sklearn import linear_model
def BayesianReg(df,z,Ncy0,Ty0):
    reg = linear_model.BayesianRidge()
    reg2 = linear_model.BayesianRidge()
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    reg.fit(X,y)
    reg2.fit(X,y2)
    predicted = reg.predict(z)
    predicted2 = reg2.predict(z)
    print('-----------------------------')
    print("Regressão de Bayesian Ridge")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2


print(" ")    
print("Regressão de Bayesian Ridge de uma composição existente")
df2 = BayesianReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 = BayesianReg(fcc_df,z,0.001,1109)

#usando SGDRegressor
from sklearn import linear_model
def SGDReg(df,z,Ncy0,Ty0):
    clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
    clf2 = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
    X = df.loc[:,'ya0':'Tx']
    y = df.loc[:,'Ty']
    y2 = df.loc[:,'Ncy']
    #z= df.loc[:,'ya0':'Tx']
    clf.fit(X,y)
    clf2.fit(X,y2)
    predicted = clf.predict(z)
    predicted2 = clf2.predict(z)
    print('-----------------------------')
    print("Regressão de SGDRegressor")
    print(print('Ty: ',predicted))
    print(print('Ncy: ',predicted2))
    
    index = []
    for ind in range(0,len(z.index)):
        index.append(ind)
    d = {'index':index ,'Ty': predicted, 'Ncy': predicted2, 'Ncy0': Ncy0, 'Ty0':Ty0}
    df2 = pd.DataFrame(data=d)
    return df2


print(" ")    
print("Regressão de SGDRegressor de uma composição existente")
df2 = SGDReg(fcc_df,fcc_df.loc[:,'ya0':'Tx'],fcc_df.loc[:,'Ncy'],fcc_df.loc[:,'Ty'])
plotarFig(df2)
df3 =SGDReg(fcc_df,z,0.001,1109)



    
