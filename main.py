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
    return 1

def phi(t):
    return t**2

#def sigma(phi, a, t,x,fbm, T): #OK, renvoie sigma(t) pour t dans [0,T]
#    n = len(x)
#    if t == 0:
#       return 0
#    else:
#        taille = int(round((t/T)*n))
#        S = np.zeros(taille)
#        for k in range(taille-1):
#            S[k] = (a(x[k]) * (fbm[k+1] - fbm[k]))
#        return phi(np.sum(S))

#def Y(sigma, t, bm, T): #OK, renvoie Y(t) pour t dans [0,T]
#    n = len(bm)
#    if t == 0:
#        return 0
#    else:
#        taille = int(round((t/T)*n))
#        S = []
#        for k in range(taille -1):
#            S.append(sigma[k] * (bm[k+1] - bm[k])) #sigma[k] = sig(kt/n)
#        return sum(S)

#def sampling(T,H,N):
#    n = 2**N
#    fbm = Fbm(H, T, n)
#    bm = Fbm(0.5, T, n)
#    x = np.linspace(0, T, n) #de la forme kT/n
#    sig = [sigma(phi,a,k/n, x , fbm, T) for k in range(n)] #de la forme k/n
#    y = []
#    for k in range(T*n):
#        y.append(Y(sig,k/n,bm,T))
#    return y

def sigma(phi, a, t, x, fbm, T):
    n = len(x)
    if t == 0:
        return 0
    else:
        taille = int(round((t / T) * n))
        fbm_diff = np.diff(fbm[:taille])  # Différence entre les valeurs successives de fbm jusqu'à la taille spécifiée
        S = a(x[:taille - 1]) * fbm_diff  # Produit élément par élément de a(x[k]) et (fbm[k+1] - fbm[k])
        return phi(np.sum(S))

def Y(sigma_values, t, bm, T):
    n = len(bm)
    if t == 0:
        return 0
    else:
        taille = int(round((t / T) * n))
        sigma_values_trimmed = sigma_values[:taille - 1][:-1]# Coupe les valeurs de sigma pour correspondre à la taille
        bm_diff = np.diff(bm[:taille-1]) # Différence entre les valeurs successives de bm jusqu'à la taille spécifiée
        return np.sum(sigma_values_trimmed * bm_diff)

def sampling(T, H, N):
    n = 2 ** N
    fbm = Fbm(H, T, n)
    bm = Fbm(0.5, T, n)
    x = np.linspace(0, T, n)  # de la forme kT/n
    sig = np.array([sigma(phi, a, k / n, x, fbm, T) for k in range(n)])  # de la forme k/n
    y = np.array([Y(sig, k / n, bm, T) for k in range(T * n)])  # Optimisé avec des opérations vectorielles
    return y


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

plt.hist(final_H(H,20,N,T)[0], range = (0,1), histtype = 'stepfilled')
plt.xlabel('valeur de H')
plt.title('Histogramme de l\'estimation de H = 0.75 pour j = 7')
plt.show()



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




