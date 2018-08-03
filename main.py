# main.py
#
# Hauptskript, in welchem der genetische Algorithmus realisiert wird.
#
# Patrick Kraus-Fuereder
# E 02.08.2018
#
# M 03.08.2018 - Erweitert auf 8 Bit
# M 03.08.2018 - Elementare Genetik

from Regressor import *
import matplotlib.pyplot as plt

ZSeed = int(floor(rand()*255))
ZSeed = fromstring(binary_repr(ZSeed), dtype='S1').astype(int)[::-1]
Ziel = array([0,0,0,0,0,0,0,0])
Ziel[0:ZSeed.size] = ZSeed
Para = 20*(rand(1,sum(Ziel))-0.5)
print(Ziel)
print(Para)

def linfun(XX):
    # Polynomfunktion, welche durch den Regressor moeglichst nahe angenaehert werden sollte.
    # Werte sollten zwischen -10 und 10 liegen.
    # P.K.-F. (02.08.2018)
    
    AIM = Ziel#array([0,1,1,0,0,1,0,0])
    Par = Para#array([1.2,-0.3,0.004])

    whe = argwhere(AIM).T
    X = zeros([XX.size,whe.size])
    for ind in range(XX.size):
        X[ind,:] = Par*XX[ind]**whe
    
    return sum(X,1)

def recombine(Genpol):
    # Diese Funktion rekombiniert die besten 10 individuen um wieder einen 
    # Genpool von 100 individuen zu schaffen.
    GENEPOOL = zeros([100,8])
    for parent_1 in range(Genpol.shape[0]):
	for parent_2 in range(Genpol.shape[0]):
	    pos = parent_1*10 + parent_2
	    for bitpos in range(Genpol.shape[1]):
		B = 0
                if Genpol[parent_1,bitpos]==Genpol[parent_2,bitpos]:
		    B = Genpol[parent_1,bitpos]
                else:
                    r = rand()
                    B = 0 if r<0.5 else 1
                # mutationsschritt
                r = rand()
                if r<0.005:
		    #print("Mutation!")
                    if B==1:
                        B = 0
                    else:
                        B = 1
		GENEPOOL[pos,bitpos] = B
    return GENEPOOL

# Initiiere ueberblicksarray ueber die Generationen
GenerationP = []

GENEPOOL = rand(100,8)
GENEPOOL[GENEPOOL<0.5] = 0
GENEPOOL[GENEPOOL>0] = 1

z = Ziel[::-1]
print("Zielgen: ",z.dot(2**arange(z.size)[::-1]))

for gen in range(10):
    if gen%1==0:
        print("Generation: ",gen)
    G_List = zeros([GENEPOOL.shape[0],1])
    
    N = 50
    X_Train = rand(N)*20-10
    Y_Train = linfun(X_Train) + rand(X_Train.size)
    
    X_Test = rand(100)*20-10
    Y_Test = linfun(X_Test)
    
    GenNow = []
    for GEN_I in range(GENEPOOL.shape[0]):
        G = GENEPOOL[GEN_I,:]
        if not sum(G)==0:
            Reg = Regressor(GENEPOOL[GEN_I,:])
            Reg.learn(X_Train,Y_Train)
            R = sum((Reg.eval(X_Test)-Y_Test)**2)
        else:
            R = 1E16
        G = G[::-1]
        GenNow.append(G.dot(2**arange(G.size)[::-1]))
        G_List[GEN_I] = R

    GenerationP.append(GenNow)

    minind = argsort(G_List.T).T
    GENEOPT = GENEPOOL[minind[0:10],:].squeeze()
    GENEPOOL = recombine(GENEOPT)

    plt.figure()
    plt.hist(GenNow)
    plt.xlim([0,256])

GenerationP = array(GenerationP)
#plt.figure("Vorher")
#plt.hist(GenerationP[0,:])
#plt.figure("Nachher")
#plt.hist(GenerationP[-1,:])
plt.show()
