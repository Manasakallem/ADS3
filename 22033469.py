

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as mpy
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import errors
import cluster_tools
from sklearn.preprocessing import normalize

"""## Clustering (K-Means)"""

cc = ['TUR','BRA','CHN','ISR']   # Selection of country codes
ind1=["EN.ATM.CO2E.KT"]   # Indicator for C02 Emission
ind1mn=['C02 Emission']
ind2=["EG.ELC.COAL.ZS"]    # indicator for Electricity production from coal source
ind2mn=['Electricity production from coal source']

my_dataframe1  = wb.data.DataFrame(ind1, cc, mrv=20).T   # read data for C02 Emission
my_dataframe1=my_dataframe1.fillna(my_dataframe1.mean())  # clean data
my_dataframe1.head()

my_dataframe2  = wb.data.DataFrame(ind2, cc, mrv=50).T   # read data for Electricity production from coal source
my_dataframe2=my_dataframe2.fillna(my_dataframe2.mean())  # clean data
my_dataframe2.head()

clmns=my_dataframe1.columns
clrs="rmby"   # asssign colours
for i in range(len(clmns)):
    mpy.figure(figsize=(6,3))   # plot figure size
    mpy.title('C02 Emission by Country for {}'.format(clmns[i]))   # plot title
    mpy.plot(my_dataframe1[clmns[i]],"{}D-".format(clrs[i]),label=clmns[i])    # line chart
    mpy.xlabel("Year")   # x-label of plotting
    mpy.xticks(rotation=90)    # x-lable rotation  
    mpy.ylabel("C02 Emission")   # y-label of plotting
    mpy.legend(loc="best")   # place legend
    mpy.grid()   # plot griding
    mpy.show()   # plot show

for i in range(len(clmns)):
    mpy.figure(figsize=(8,3))   # plot figure size
    mpy.title('Electricity production from coal source for {}'.format(clmns[i]))   # plot title
    mpy.plot(my_dataframe2[clmns[i]],"{}D-".format(clrs[i]),label=clmns[i])      # line chart
    mpy.xlabel("Year")   # x-label of plotting
    mpy.xticks(rotation=90)    # x-lable rotation
    mpy.ylabel("Electricity production")   # y-label of plotting
    mpy.legend(loc="best")   # place legend
    mpy.grid()   # plot griding
    mpy.show()   # plot show

def corrmapviz(dt):   # to visualize country correlation with indicators
    cluster_tools.map_corr(dt)    # call process
dfs=[my_dataframe1,my_dataframe2]    # accumulatre dataframes
corrmapviz(dfs[0])    # visualize correlation
corrmapviz(dfs[1])    # visualize correlation

def nrmdata(df):   # method to call normalization process
    nrmouts=cluster_tools.scaler(df)    # call process
    return nrmouts[0], nrmouts[1], nrmouts[2]
scldfs=[]
mnvls=[]
mxvls=[]
for d in range(len(dfs)):
    outnrm=nrmdata(dfs[d])   # calling method
    scldfs.append(outnrm[0])    # storing scaled(normalized) data
    mnvls.append(outnrm[1])     # storing minimum value
    mxvls.append(outnrm[2])     # storing maximum value
print(scldfs[0].head(),"\n")
print(scldfs[1].head())

def optclus(dt):   # selection of optimum cluster value
    optclus_ws = []
    val=10
    for i in range(1, val):   # K-Means clustering with cluster 1 to 10
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=600,  random_state=32)   # create K-means model
        kmeans.fit(dt)    # train k-means model
        optclus_ws.append(kmeans.inertia_)   # Storing inertia vaklues for all clusters
    arrws=np.array(optclus_ws)
    arrws=arrws[arrws>1]    # Finding the value after which elbow curve smooths
    clopt=arrws[-1]    # Finding optimum value of cluster
    optclus=optclus_ws.index(clopt)
    return optclus, optclus_ws, val

clus, allws, clsvl=optclus(scldfs[0])  # 
mpy.figure(figsize=(5,3))   # plot figure size
mpy.title('Elbow Curve (Optimum Cluster: {})'.format(clus))   # plot title
mpy.plot(range(1, clsvl), allws,"c--")    # plot inertia value
mpy.plot(range(1, clsvl), allws,"Xm")
mpy.xlabel('Number of clusters')   # x-label of plotting
mpy.ylabel('Inertia')   # y-label of plotting
mpy.grid()   # plot griding
mpy.show()   # plot show

kmeans = KMeans(n_clusters=clus, init='k-means++', max_iter=300, n_init=10, random_state=0)   # final kmeans with optrimum cluster
kmd = kmeans.fit(scldfs[0])   # train kmeans
print("Cluster Centres:",kmd.cluster_centers_)

kmd.cluster_centers_

allcntr=[]
for i in kmd.labels_:
    if i==0:
        allcntr.append(clmns[0])
    elif i==1:
        allcntr.append(clmns[1])
    elif i==2:
        allcntr.append(clmns[2])
    elif i==3:
        allcntr.append(clmns[3])
    elif i==4:
        allcntr.append(clmns[4])
    else:
        pass

df=pd.DataFrame(scldfs[0],columns=my_dataframe1.columns)
mpy.figure(figsize=(6,3))   # plot figure size
mpy.title('Cluster Visualization')   # plot title
sns.scatterplot(data=df, x=clmns[0], y=clmns[1], hue=allcntr,palette="PuRd")   # scatter plot for clsuter visualization
mpy.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="d", c="b", s=80, label="centroids")
mpy.legend()  # place legend
mpy.grid()    # plot griding
mpy.show()   # plot show

"""## Curve Fitting"""

from scipy.optimize import curve_fit
#!pip install lmfit
from lmfit import Model

def func(x, amp, cen, wid):   # method for curve fitting
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**3 / (2*wid**2))

norml2 = nrmdata(my_dataframe2.values)   # normalize data
y = func(scldfs[1].iloc[:,1], 2.1, 0.7, 1.51) + np.random.normal(0, 0.2, norml2[0].shape[0])   # calling method
init_vals = [2, 0, 2] 
best_vals, covar = curve_fit(func, norml2[0][:,1], y, p0=init_vals,maxfev = 700)   # curve fitting
gmodel = Model(func)  # preparing curve fitting

result = gmodel.fit(y, x=norml2[0][:,1], amp=5, cen=3, wid=0.4)   # train model;
plt.figure(figsize=(6,4))   # plot figure size
plt.title('Curve Fitting Result')   # plot title
plt.plot(norml2[0][:,1],"bo",label="Data")
plt.plot(result.init_fit, 'm--', label='Initial fit')
plt.plot(result.best_fit, 'c-', label='Best fit')
plt.legend()  # place legend
plt.grid()    # plot griding
plt.show()   # plot show

result

