# main.py
#
# Hauptskript, in welchem der genetische Algorithmus realisiert wird.
#
# Patrick Kraus-Fuereder
# 02.08.2018

from Regressor import *
from numpy import NaN

def linfun(XX):
    # Polynomfunktion, welche durch den Regressor moeglichst nahe angenaehert werden sollte.
    # Werte sollten zwischen -10 und 10 liegen.
    # P.K.-F. (02.08.2018)
    
    AIM = array([0,1,1,0,0,1])
    Par = array([1.2,-0.3,0.004])

    whe = argwhere(AIM).T
    X = zeros([XX.size,whe.size])
    for ind in range(XX.size):
        X[ind,:] = Par*XX[ind]**whe
    
    return sum(X,1)



GENEPOOL = rand(300,6)
GENEPOOL[GENEPOOL<0.5] = 0
GENEPOOL[GENEPOOL>0] = 1

G_List = zeros([GENEPOOL.shape[0],1])

N = 1000
X_Train = rand(N)*20-10
Y_Train = linfun(X_Train) + rand(X_Train.size)

X_Test = rand(100)*20-10
Y_Test = linfun(X_Test)

for GEN_I in range(GENEPOOL.shape[0]):
    print(GENEPOOL[GEN_I,:])
    if not sum(GENEPOOL[GEN_I,:])==0:
        Reg = Regressor(GENEPOOL[GEN_I,:])
        Reg.learn(X_Train,Y_Train)
        R = sum((Reg.eval(X_Test)-Y_Test)**2)
    else:
        R = 1E16
    G_List[GEN_I] = R

minind = argmin(G_List)
print(G_List[minind])
print(GENEPOOL[minind,:])
