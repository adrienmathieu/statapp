def read_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = [float(row[0]) for row in reader]
    return data

#fBM = read_csv('valeurs_fBM2.csv')
#bM = read_csv('valeurs_BM2.csv')


from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt
import csv

def Fbm(H, T, n):
    f = FBM(n, hurst=H, length=T)
    fbm_sample = list(f.fbm()[:-1])
    return fbm_sample

def a(t):
    return t #faut trouver une autre fonction mais en sah ça marche

def phi(t):
    return t**2

def sigma(phi, a, x, fbm, T): #marche
    a_values = a(x)[:-1]
    fbm_diff = np.diff(fbm)
    a_dot_f = a_values*fbm_diff
    cum_a_dot_t = np.cumsum(a_dot_f)
    SIGMA = [0]+[phi(cum_a_dot_t[i*64]) for i in range(1,T*64*n)]
    return SIGMA

T = 3
H = 0.1
N = 10
n = 2**N
sample = 64
x = np.linspace(0,3,sample*sample*T*n)

def Y_sampling(phi,a,bm,T): #marche
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

def final_H(H,sample,N,T):
    h = []
    j = 6
    for i in range(sample):
        y = sampling(T,H,N)
        Z = z(y)
        h.append(H_estim(j,y,Z))
    return (h,sum(h)/sample)

T = 3
H = 0.75
N = 12
n = 2**N
#Y_sampling = read_csv('valeurs_y.csv')
#Z = z(Y_sampling)
# we observe Y_{i/2**N} for i in [0,T*2**N]



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




