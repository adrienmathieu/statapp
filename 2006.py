from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt
import csv


def read_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = [float(row[0]) for row in reader]
    return data

def Fbm(H, T, n):
    f = FBM(n, hurst=H, length=T)
    fbm_sample = list(f.fbm()[:-1])
    return fbm_sample

def a(t):
    return t

def phi(t):
    return t**2

def sigma(phi, a, x, fbm, T): #marche
    a_values = a(x)[:-1]
    fbm_diff = np.diff(fbm)
    a_dot_f = a_values*fbm_diff
    cum_a_dot_t = np.cumsum(a_dot_f)
    SIGMA = [0]+[phi(cum_a_dot_t[i*64]) for i in range(1,T*64*n)]
    return SIGMA


def Y_sampling(phi,a,bm,T, fBM): #marche
    bm_diff = np.diff(bm)
    Sig = sigma(phi,a,x,fBM,T)[:-1]
    a_dot_f = Sig*bm_diff
    cum_a_dot_t = np.cumsum(a_dot_f)
    Y  = [0] + [cum_a_dot_t[i*64] for i in  range(1,T*n)]
    return Y

def z(Y):
    length = len(Y)
    n = int(length/T)
    Z = []
    for l in range(length-1):
        Z.append(n*(Y[l+1] - Y[l])**2)
    return Z

def integral(k,l,j,N):
    sum = 0
    if  (2*k+1)/(2**(j+1)) <= k/(2**j) + (l+1)/(2**N) and 1/(2**(j+1)) > l/(2**N):
        sum +=(1/(2**(j+1))-l/(2**N))
    elif  (2*k+1)/(2**(j+1)) > k/(2**j) + (l+1)/(2**N):
        sum+= 1/(2**N)
    elif (2*k+1)/(2**(j+1)) <  k/(2**j) + (l)/(2**N):
        sum+=0
    if  (2*k+2)/(2**(j+1)) <= k/(2**j) + (l+1)/(2**N) and (2*k+1)/(2**(j+1)) <= k/(2**j) + l/(2**N):
        sum -= ((2*k+2)/(2**(j+1)) - (k/(2**j) + l/(2**N)))
    elif  (2*k+2)/(2**(j+1)) > k/(2**j) + (l+1)/(2**N) and (2*k+1)/(2**(j+1)) <= k/(2**j) + l/(2**N):
        sum -= (1/(2**N))
    elif (2*k+2)/(2**(j+1)) > k/(2**j) + (l+1)/(2**N) and (2*k+1)/(2**(j+1)) >k/(2**j) + l/(2**N):
        sum -= ( k/(2**j) + (l+1)/(2**N)-(2*k+1)/(2**(j+1)))
    elif (2*k+2)/(2**(j+1)) < k/(2**j) + (l)/(2**N):
        sum-=0
    elif (2*k+2)/(2**(j+1)) <= k/(2**j) + (l+1)/(2**N) and (2*k+1)/(2**(j+1)) > k/(2**j) + l/(2**N):
        sum -= 1/(2**(j+1))
    elif (2*k+1)/(2**(j+1)) > k/(2**j) + (l+1)/(2**N):
        sum-=0
    return 2**(j/2)*sum

def estim_d(j,k,z):
    D = [z[k*(2**(N-j))+l]*integral(k,l,j,N) for l in range(T*2**(N-j)-1)]
    return sum(D)

def estim_a(j,k,l,Y):
    h_n = round(np.sqrt(n))
    s = [(Y[2**(N-j)*k+(l+p+1)] - Y[k*2**(N-j)+(l+p)])**2 for p in range(1,h_n +1)]
    return ((np.sqrt(2)/h_n * sum(s))**2)

def estim_nu(j,k,Y):
    S = [integral(k,l,j,N)**2 * estim_a(j,k,l,Y) for l in range(T*2**(N-j)-1)]
    return (n**2*sum(S))

def d_final(k,j,Y,Z):
    return estim_d(j,k,Z)**2 - estim_nu(j,k,Y)

def estim_Q(j,Y,Z):
    A =[d_final(k,j,Y,Z) for k in range(int(-T-2**(j-N/2)), int(T*(2**j -1) - 2**(j-N/2)))]
    return sum(A)

def H_estim(j,Y,Z):
    return -1/2*(np.log(estim_Q(j+1,Y,Z)/estim_Q(j,Y,Z))/np.log(2))

def _pre_final_H(H,index):
    h = []
    j = 3
    for i in range(index):
        BM = Fbm(0.5,T,sample*n*T)
        fbm = Fbm(H,T,sample*sample*T*n)
        y = Y_sampling(phi,a,BM,T,fbm)
        Z = z(y)
        h.append(H_estim(j,y,Z))
    return (h, sum(h)/index)


#c'est ça qu'on fait tourner :
T = 3
H = 0.8 #j = 3 ça marche bien pour H = 0.7
N = 10
n = 2**N
sample = 64
index = 10
x = np.linspace(0,T,sample*sample*T*n)
print(_pre_final_H(H,index))






def calcul(): #ça c'est si on veut juste faire tourner avec les doc .csv
    fBM = read_csv('valeurs_gros_fbm.csv')    #taille : sample*sample*T*n
    bM = read_csv('valeurs_gros_bm.csv')     #taille : sample*T*n
    H = []
    for k in range(50):
        x = np.linspace(0,T,sample*sample*T*n)
        BM = Fbm(0.5,T,sample*n*T)
        Ysampling = Y_sampling(phi,a,BM,T,fBM)
        Zsampling = z(Ysampling)
        H.append(H_estim(3,Ysampling,Zsampling))
    return (H,sum(H)/50)



####### DIFFERENTES VALEURS TESTÉES (H : valeur de H, H_my : moyenne des valeurs estimées)
H =[0.6909249820048948, 0.5421058612013202, 0.644941946787065, 0.6009755920092263, 0.6341611000189057, 0.7255671884439706,
    0.5797747284052278, 0.7026264076632935, 0.5497029012035453, 0.8366766981391411, 0.7735541673987709, 0.7164528646724577,
    0.5271688093461537, 0.7687714841929163, 0.5524787804765722, 0.760441519467048, 0.6044956713714713, 0.6813347442777907,
    0.5879062085491543, 0.6994248092864387, 0.5577949583077234, 0.66334288395691, 0.5371006958291066, 0.6509999339917816,
    0.8436012885524339, 0.6369397010021566, 0.5466127479681444, 0.690477069172943, 0.6596740800455565, 0.6426031173095285,
    0.6282072263415907, 0.8619445326888311, 0.6203081933266542, 0.6146610748375256, 0.6955086899542529, 0.7108468073230073,
    0.5495653268175226, 0.6726085244736343, 0.7923757252923613, 1.1493693107291922, 0.6054794066608523, 0.43530169229845084,
    0.685683455907876, 0.5555612084846877, 0.672750002949368, 0.5937445976447263, 0.831857662712069, 0.5679635131991349,
    0.7955323623906772, 0.49242405065819433] # H = 0.7, H_my = 0.66

I = [-0.7487444990350632, 0.3510945401283077, 0.9056514998765163, 1.0042565462327693, 1.1185673363940558, 0.6399243153364972,
     0.575301797818461, -0.6208785400195914, 1.0469172868954544, 0.3002782100074318] # H = 0.55, H_my = 0.45

J = [0.8647930957225007, 1.001961302326304, 0.8668024953940235, 0.5555305296069277, 1.0323026090682867, 0.3497630074498283,
     0.9363363713734275, 0.6292499667790874, 0.7279337110344142, 0.777008541218553] # H = 0.3, H_my = 0.77

K = [-1.1422869846495138, 0.9168715284228759, 0.0761189795068077, 0.7800249883675477, 0.8405012091893524, 0.734395298381343,
     0.8690178050825297, 0.739346280366979, 0.5291610843376251, 0.7126113362959711] # H = 0.5, H_my = 0.505

L = [0.5556104526573825, 0.5400762624338373, 0.9677718131890833, 1.0193058331465332, 0.9316892980063575, 1.1453472640039675,
     0.6310209273146726, 1.0387873024266747, 0.8931847849897481, -0.08175874571591993], #H = 0.8, H_my = 0.7641035192452337


############################################
############### on test pour plusieurs valeurs de H :

def d_haar(j,k,x,fbm, N,T = 5):
    return 2**(-j)*(sigma(phi,a,(k+T)/(2**j),N, x,fbm, T)**2-sigma(phi,a,k/(2**j),N, x,fbm, T)**2)**2

def Q(j,x,fbm, N):
    V = [d_haar(j,k, x, N, fbm) for k in range(100)]
    #range(((2 ** j) - 1) * T)]
    return sum(V)

def val(j, x, fbm,N):
    return (-1/2*(np.log(Q(j,x, fbm,N)/Q(j-1, x,fbm,N))/np.log(2)))

def J_star(n,x,bm,fbm):
    q = []
    for j in range(1,round(n/2)):
        q.append(Q(j,x,bm,fbm)-(2**j)/n)

    indice = None
    for i in range(len(q)):
        if q[i] > 0:
            indice = i
    return indice, q

#j = 4 for H = 0.8
#j = 5 for H = 0.5
#j = 4 for H = 0.9
#j = 5 for H = 0.6

def final_h(H,sample):
    h = []
    n = 2**15
    T = 5
    x = np.linspace(0, T, n)
    bm = Fbm(0.5, T, n)
    j = 5
    for i in range(sample):
        fbm = Fbm(H, T, n)
        h.append(val(j, x, fbm, n))
    return h
###############################################
###############################################







