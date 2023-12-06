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
        np.round(x, 3)
        x[:,i+1]= x[:,i] + np.multiply(3/2*h,func(t[i], x[:,i])) - np.multiply(1/2*h,func(t[i-1], x[:,i-1]))
    return x

#Runge-Kutta 4th Order
def RK4(func, h, xn, tn):
    k1 = func(tn, xn)
    k2 = func((tn + h/2), (xn + np.multiply(k1,h/2)))
    k3 = func((tn + h/2), (xn + np.multiply(k2,h/2)))
    k4 = func((tn + h), (xn + np.multiply(k3, h)))
    k = np.round([k1, k2, k3, k4], 6).reshape(3,4)
    return xn + np.multiply(h/6,k[:,0] + np.multiply(2,k[:,1]) + np.multiply(2,k[:,2]) + k[:,3])

#ODE Functions
def helix(t, xi):
    w = 1
    k = 2
    x = -w*xi[1]
    y = w*xi[0]
    return [x, y, k]

def Lorenz(t, xi):
    sig = 10
    p = 28
    beta = 8/3
    x = sig*(xi[1]-xi[0])
    y = xi[0]*(p - xi[2])
    z = xi[0]*xi[1] - beta*xi[2]
    return [x, y, z]

#True Functions
def checkhelix(h, N, x0, tn_1):
    x = np.zeros(shape=(3, N+1), dtype=np.float128)
    t = np.arange(tn_1, (N*h)+tn_1, h)
    x[:,0]=x0
    for i in range(0,N):
        x[:,i+1] = [3*np.cos(t[i]), 3*np.sin(t[i]), (2*t[i])]
    return x

#Calculators with Graphs (for testing only)
def helix_demo():
    pred_hel = AB2_3D(func=helix, h=.1, N=500, x0=[3, 0, 0], x1=[1.62, 2.52, 2], tn_1=1)
    tru_hel = checkhelix(h=.1, N=500, x0=[3, 0, 0], tn_1=1)

    fig, ax = plt.subplots(1,2, subplot_kw={'projection': "3d"})
    ax[0].plot(pred_hel[0,:], pred_hel[1,:], pred_hel[2,:])
    ax[1].plot(tru_hel[0,:], tru_hel[1,:], tru_hel[2,:])
    plt.show()

def lorenz_calc():
    pred_lor = AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1,1,1], x1=RK4(Lorenz, h=.0025, xn=[1,1,1], tn=0), tn_1=0)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pred_lor[0,:], pred_lor[1,:], pred_lor[2,:], lw=.3)
    plt.show()
    

if __name__ == '__main__':

    pred_lor = [AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1,1,1], x1=RK4(Lorenz, h=.0025, xn=[1,1,1], tn=0), tn_1=0),
                AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1.01,1,1], x1=RK4(Lorenz, h=.0025, xn=[1.1,1,1], tn=0), tn_1=0),
                AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1.3,1,1], x1=RK4(Lorenz, h=.0025, xn=[2,1,1], tn=0), tn_1=0)]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle('Lorenz System Through AB2 and RK4')
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1.set_title('x0 = [1,1,1]')
    ax1.plot(pred_lor[0][0,:], pred_lor[0][1,:], pred_lor[0][2,:], lw=.3)
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax2.set_title('x0 = [1.01,1,1]')
    ax2.plot(pred_lor[1][0,:], pred_lor[1][1,:], pred_lor[1][2,:], lw=.3)
    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3.set_title('x0 = [1.3,1,1]')
    ax3.plot(pred_lor[2][0,:], pred_lor[2][1,:], pred_lor[2][2,:], lw=.3)

    plt.show()
