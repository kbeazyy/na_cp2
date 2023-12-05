import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

#Adams Bashforth 2, 3 Dimensions
#params:
    #func: ODE
    #h: step size
    #N: number of steps
    #x0, x1: necessary history(true values or estimation)
    #tn_1: t sub n-1, starting time
def AB2_3D(func, h, N, x0, x1, tn_1):
    x = np.zeros(shape=(3,N+1))
    t = np.arange(tn_1, (N*h)+tn_1, h)
    x[:,0]=x0
    x[:,1]=x1
    for i in range(1,N):
        x[:,i+1]= x[:,i] + np.multiply(3/2*h,func(t[i], x[:,i])) - np.multiply(1/2*h,func(t[i-1], x[:,i-1]))
    return x

#ODE Functions
def helix(t, xi):
    w = 1
    k = 2
    x = -w*xi[1]
    y = w*xi[0]
    return [x, y, k]

#True Functions
def checkhelix(h, N, x0, tn_1):
    x = np.zeros(shape=(3, N+1))
    t = np.arange(tn_1, (N*h)+tn_1, h)
    x[:,0]=x0
    for i in range(0,N):
        x[:,i+1] = [3*np.cos(t[i]), 3*np.sin(t[i]), (2*t[i])]
    return x

#Calculators with Graphs
def helix_demo():
    pred_hel = AB2_3D(func=helix, h=.1, N=500, x0=[3, 0, 0], x1=[1.62, 2.52, 2], tn_1=1)
    tru_hel = checkhelix(h=.1, N=500, x0=[3, 0, 0], tn_1=1)

    fig, ax = plt.subplots(1,2, subplot_kw={'projection': "3d"})
    ax[0].plot(pred_hel[0,:], pred_hel[1,:], pred_hel[2,:])
    ax[1].plot(tru_hel[0,:], tru_hel[1,:], tru_hel[2,:])
    plt.show()

