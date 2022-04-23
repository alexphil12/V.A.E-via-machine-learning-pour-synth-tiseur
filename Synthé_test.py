
"""
@author: Alexandre Philippon
"""
import numpy as np#The User guide is on the bottom of this document
from scipy import signal
from math import floor
import random as rd
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
def synthe_themis(freq, switch,PWM,A,B,ordre,F0,Q,ADSR):
    fe=44100;#this algorithm aim to create a signal of 4 s that could be played on the themis
    T=np.linspace(0,4,4*fe)
    sawtooh_A= signal.sawtooth(2*np.pi*freq*T)#this part generate the signal that will be the component of the audio signal 
    triangle_A=signal.sawtooth(2*np.pi*freq*T,width=0.5)#they generate the signal which they are named after.
    Pulse_A=signal.square(2*np.pi*T*freq,duty=PWM)#the number of cases in this list(176400)is also the basic number of features.   
    sawtooh_B= signal.sawtooth(2*np.pi*freq*T)#this step generates the out of the VCOs
    triangle_B=signal.sawtooth(2*np.pi*freq*T,width=0.5)
    Pulse_B=signal.square(2*np.pi*T*freq,duty=PWM)   
    subass=signal.square(np.pi*freq*T)
    carrée=signal.square(2*np.pi*T*freq)
    triangle2=signal.sawtooth(2*np.pi*freq*T,width=0.5)
    if switch==0:
        sig_A=sawtooh_A
    elif switch==1:
        sig_A=triangle_A#We make here the analog switch selection
    else:
        sig_A=Pulse_A            
    Signal_audio=(A[0]*triangle_B+A[1]*sig_A+A[2]*sawtooh_B+A[3]*Pulse_B+B[0]*carrée+B[1]*triangle2+B[2]*subass)#the signal are summing into one signal, that's here the mixers out
    Spectre_signal=np.fft.fft(Signal_audio)/len(Signal_audio)
    Spectre_signal[0]=0#We don't need any mean terms
    Spectre_signal[-1]=0
    X_filtre=np.linspace(0,fe/2,2*fe)
    i=complex(0,1)
    filtre= lambda f:1/(1+i*(f/(F0*Q))-((f/F0)**2))#the basic transfert function of a low pass filter
    if ordre==2:
        filt_produit_1=[filtre(x) for x in X_filtre]
        filt_produit_2=filt_produit_1[::-1] #on this part we create a list with is filter([0:fe/2]) we create then it's symetric (or reverse) list and concatenate the two,this way
        filt_produit=np.concatenate((filt_produit_1,filt_produit_2),axis=None)#when the two list(filt_produit and Spectre_filtre) ar multiply term by term, the result is the spectrum of a filtered signal.
        Spectre_filtre=filt_produit*Spectre_signal
    if  ordre==4:#the order 4 filter is always the square of the order 2
        filt_produit_1=[(filtre(x)**2) for x in X_filtre]
        filt_produit_2=filt_produit_1[::-1]
        filt_produit=np.concatenate((filt_produit_1,filt_produit_2),axis=None)
        Spectre_filtre=filt_produit*Spectre_signal
    Signal_filtre=np.fft.ifft(Spectre_filtre)*len(Signal_audio)#We recreate the signal with an inverse fft,(note that the result is complex but we are only looking for the real part )
    Tmonte=ADSR[0]
    Ts=ADSR[1]
    N1=floor(Tmonte*fe)
    N2=floor(Ts*fe)
    coef1=(1-0)/(N1-0)#this part aim to calculates the enveloppe with the ADSR coefficient 
    coef2=-coef1
    envellope=[0]
    k=0
    pente1=lambda k:coef1*k
    pente2=lambda k:coef2*k+2
    pente3=lambda k:coef2*(k-(N1+N1//2+N2))+0.5
    for k in range(1,N1+1):
        envellope.append(pente1(k))
    for k in range(N1+1,N1+1+N1//2):
        envellope.append(pente2(k))
    for k in range(1,N2+1):
        envellope.append(0.5)
    for k in range(N1+N1//2+N2,2*N1+N2+1):
        envellope.append(pente3(k))
    while(len(envellope)<len(Signal_filtre)):
        envellope.append(0)
    Signal_final=Signal_filtre*envellope/max(Signal_filtre)#We multiply the enveloppe and the filtered signal to obtain the final signal that we then normalize
    Signal_final=Signal_final/max(Signal_final)
    return(Signal_final.real)

def data_set_themis(N):
    data_set=[]#this algorithm chooses randomly parameters,creates the signal associated and return a list of N sample with the parameters+ the signal in one case 
    Freq_tab=[43.654*2**(k/12) for k in range(0,73)]#this table of frequency is based on the Bach tonal equalization and the the note the themis can produce
    Switch_tab=[0,1,2]
    ordre_tab=[2,4]
    for U in range(1,N+1):
        A=[]
        B=[]
        ADSR=[]
        freq=rd.choice(Freq_tab)#we choose randomly an element of Freq_tab
        for p in range(0,4):
            A.append(rd.random())
        for m in range(0,3):
            B.append(rd.random())
        switch=rd.choice(Switch_tab)
        PWM=rd.random()
        ordre=rd.choice(ordre_tab)
        F0=rd.uniform(freq,20000)#it's useless to filter bellow the Fundamental frequency quite obviously and over 20000hz because Fe~=40000
        Q=rd.uniform(0,3)#With this configuration we will have a little more case with resonnance which is in my opinion more common 
        ADSR.append(rd.uniform(0,1.5))
        ADSR.append(4-2*ADSR[0])#The two line(97_98) allows that all signal will have an entire second of silence,I choose to do so to make the work easier when the models will be integrated in the themis.
        data_set.append([freq, switch,PWM,A,B,ordre,F0,Q,ADSR,synthe_themis(freq, switch,PWM,A,B,ordre,F0,Q,ADSR)])
    return(data_set)    
def data_set_formation_regression(data_set):
    fe=44100#this algorith aims to create a data-set well create for the regression problem(the main idea is to reduce the number of features)
    test_set=[]#it's supposed to work on a data_set of 2000 samples.If you want to change the number of sample.
    vrai_etat=[]#You'll have to change the number on the "for" loop.
    train_set=[]#it return a list with four element:test_set and vrai_etat contenain the features and the label for the test set
    info_set=[]#train set and info_set contain the features and the label of the train set
    for k in range(0,1800):
        ech=data_set[k][9][floor(fe*1.5*data_set[k][8][0]):floor(1000+fe*1.5*data_set[k][8][0])]#here ech contain the first 1000 features of the sustain phase of the signal
        ech1=ech.tolist()
        ech1=ech1*10#we copy this list 10 time and concatenante the result
        ech1=np.asarray(ech1)
        ech1=np.multiply(ech,1/max(ech))#normalization
        train_set.append(ech1)#info set get all the label of the signal k,the triple [] is because the data A and B are also list and that the algorithm of machine learning can not have a list as an out
        info_set.append([data_set[k][2],data_set[k][3][0],data_set[k][3][1],data_set[k][3][2],data_set[k][3][3],data_set[k][4][0],data_set[k][4][1],data_set[k][4][2],data_set[k][6],data_set[k][7]])
    for j in range(1800,2000):
        ech_test=data_set[j][9][floor(fe*1.5*data_set[j][8][0]):floor(1000+fe*1.5*data_set[j][8][0])]
        ech_test1=ech_test.tolist()
        ech_test1=ech_test1*10
        ech_test1=np.asarray(ech_test1)
        ech_test=np.multiply(ech_test,1/max(ech_test))
        test_set.append(ech_test)
        vrai_etat.append([data_set[j][2],data_set[j][3][0],data_set[j][3][1],data_set[j][3][2],data_set[j][3][3],data_set[j][4][0],data_set[j][4][1],data_set[j][4][2],data_set[j][6],data_set[j][7]])
    return([train_set,info_set,test_set,vrai_etat])
def data_set_formation_classification(data_set):
    fe=fe=44100#this algorithm does quite the same thing as data_set_formation_regression with only three differences.
    data_set_clas=[]#1 he makes the fft off the ech and then only keep half of it
    Y_data_clas=[]#2 He can take any size of data-set
    validation=[]#3 he creates a validation set but no test-set, this one is created in classification themis
    data_validation=[]
    for k in range(0,floor(len(data_set)*0.9)):
        ech=data_set[k][9][floor(fe*1.5*data_set[k][8][0]):floor(1000+fe*1.5*data_set[k][8][0])]
        ech1=ech.tolist()
        ech1=ech1*10
        ech1=abs(np.fft.fft(ech1))
        ech1[0]=0
        ech1[-1]=0
        ech1=np.asarray(ech1)
        ech1=np.multiply(ech1,1/max(ech1))
        ech1=ech1[0:len(ech1)//2]
        data_set_clas.append(ech1)
        Y_data_clas.append((data_set[k][1],data_set[k][5])) 
    for j in range(floor(len(data_set)*0.9),len(data_set)):
       ech_va=data_set[j][9][floor(fe*1.5*data_set[j][8][0]):floor(1000+fe*1.5*data_set[j][8][0])]
       ech_va1=ech_va.tolist()
       ech_va1=ech_va1*10
       ech_va1=abs(np.fft.fft(ech_va1))
       ech_va1[0]=0
       ech_va1[-1]=0
       ech_va1=np.asarray(ech_va1)
       ech_va1=np.multiply(ech_va1,1/max(ech_va1))
       ech_va1=ech_va1[0:len(ech_va1)//2]
       validation.append(ech_va1)     
       data_validation.append((data_set[j][1],data_set[j][5])) 
    return([data_set_clas,Y_data_clas,validation,data_validation])

def classification_themis(data_sets):
    X_or=np.asarray(data_sets[0])#this algorithm creates the models and make the "machine learning classification" process
    X_sw=np.asarray(data_sets[0])#this algorithm return actualy nothing, if you want to consult for example the score of a model
    data=np.asarray(data_sets[1])#you can add on the end return(model.score or model2.score)
    y_or=data[:,1]
    y_sw=data[:,0]
    validation=np.asarray(data_sets[2])
    data_val=np.asarray(data_sets[3])
    ordre_val=data_val[:,1]
    switch_val=data_val[:,0]
    X_train_or,X_test_or,y_train_or,y_test_or=train_test_split(X_or,y_or,test_size=0.2)
    X_train_sw,X_test_sw,y_train_sw,y_test_sw=train_test_split(X_sw,y_sw,test_size=0.2)
                                                               
    model=KNeighborsClassifier()
    model2=KNeighborsClassifier()
    model.fit(X_train_or,y_train_or)
    model2.fit(X_train_sw,y_train_sw)
    k=np.arange(1,20)
    
def regression_themis(data_sets):
    X=np.asarray(data_sets[0])#Same principle that the one before. But here all the set have been already done
    Y=data_sets[1]#it suppose to return the mean of correlation vector for all the task (PWM,A[1],B[2]etcc)
    test_set=np.asarray(data_sets[2])
    Y_test=np.asarray(data_sets[3])
    
    model= MultiTaskLasso(alpha=1,max_iter=10000)
    model.fit(X,Y)
    predi=model.predict(test_set) 
    return(r2_score(Y_test,model.predict(test_set)))

def retourn_fréquence(signal):
    fe=44100
    H=abs(np.fft.fft(signal))/len(signal)
    H=H[0:len(H)//2]
    H[0]=0
    Q=max(H)
    count=0
    while H[count]!=Q:
        count+=1;
    N=len(H)
    freq=(fe/(2*N))*count
    return(freq)

def retourn_ADSR(Signal):
    fe=44100#this algorithm only work for a 4 seconds sample.
    Q=max(Signal)
    ADSR=[0,0]
    count=0
    while Signal[count]!=Q:
        count+=1
    ADSR[0]=count*(1/fe)    
        
    count2=176400
    while Signal[count2-1]==0:
        count2-=1
    T0=(176400-count2-1)/fe
    ADSR[1]=4-(ADSR[0]*2+T0)
    return(ADSR)    
#User guide:      
#To make a simulation and develope a model of machine learning
#   press F5 to execute the script or if you use spyder press with your mouse the green triangle on top left of your screen
#   1:write on the consol:"data_set=data_set_themis(N)" where N is the number of sample you want.Be carrefull over an hundred            
#   the algorithm can take a lot of time(in my case my computer needed a whole night to make a 2000 sample data_set)     
#   2:write on the consol:"data_set_regression=data_set_formation_regression(data_set)" or "data_set_classification=data_set_formation_classification(data_set)"
#   at this stage you will obtain either a list of 4 element.        
#   3:write on the consol:"regression_themis(data_set_regression)"or "classification_themis(data_set_classification)"
#   the result will depend on what you want to see. for further information,I suggest you to consult the sci kit learn
#   user guide and also the documentation about model.
#   If you are a french speaker I also suggest you to watch the Youtube channel machine learnia wich explain really well      
#   how to use sci kit learn object 
#        
#   PS:If you want to plot some signal you should write on the consol:"H=synthe_themis(with the parameters you want)     
#   and then write(plt.plot) 
#    
#
#    
#    