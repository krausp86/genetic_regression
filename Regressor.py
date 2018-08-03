# Regressor.py
#
# Klassendefinitionsskript fuer das genetisch abhaengige Regressor-Objekt
#
# Patrick Kraus-Fuereder
# 02.08.2018

from scipy import *
from scipy import linalg

class Regressor:
    # Regressor-Klasse welche auf Basis eines "Gencodes" eine gegebene Sequenz fittet
    
    def __init__(self,gen):
        self.genome = gen
        self.params = zeros([6,1])
        self.fit_avail = False

    def learn(self,X_Train,Y_Train):
        C = zeros([X_Train.size,int(self.genome.sum())])
        for ind in range(X_Train.size):
            C[ind,:] = X_Train[ind]**argwhere(self.genome).T
        
        # Nun versuche die Y_Train-Werte zu fitten
        k = dot(dot(linalg.inv( dot( C.T, C ) ),C.T),Y_Train)
	self.params = k

    	# Die Parameterlisten sollten jetzt vollstaendig sein, setze fitflag auf true
	self.fit_avail = True

    def eval(self,X_Test):
	# Zur Evaluation wird fast das gleiche getan, Es wird eine Matrix C
	# erstellt, welche so viele Zeilen hat wie X_Test elemente und so viele
	# Spalten wie parameter gefittet wurden.
	#
	# Sollte allerdings noch kein Fit erfolgt sein so kann nicht evaluiert werden

	if not self.fit_avail:
	    return None
	else:
            if sum(self.genome)==0:
                return zeros(X_Test.shape)
	    else:
     	        pos = argwhere(self.genome)
	        C = zeros([X_Test.size,pos.size])
	        for ind in range(X_Test.size):
		    C[ind,:] = X_Test[ind]**pos.T * self.params
                
	        Y_Test = sum(C,1)
                return Y_Test

