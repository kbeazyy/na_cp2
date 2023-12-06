import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)


def AB2_3D(func, h, N, x0, x1, tn_1):
    """
    Adams Bashforth 2, 3 Dimensions

    Parameters:
        func (function): ODE function
        h (float): step size
        N (int): number of steps
        x0 (list): necessary history (true values or estimation)
        x1 (list): necessary history (true values or estimation)
        tn_1 (float): starting time

    Returns:
        numpy.ndarray: Array of shape (3, N+1) containing the solution
    """
    x = np.zeros(shape=(3,N+1))
    t = np.arange(tn_1, (N*h)+tn_1, h)
    x[:,0]=x0
    x[:,1]=x1
    for i in range(1,N):
        np.round(x, 3)
        x[:,i+1]= x[:,i] + np.multiply(3/2*h,func(t[i], x[:,i])) - np.multiply(1/2*h,func(t[i-1], x[:,i-1]))
    return x


def RK4(func, h, xn, tn):
    """
    Runge-Kutta 4th Order

    Parameters:
        func (function): ODE function
        h (float): step size
        xn (list): current state
        tn (float): current time

    Returns:
        numpy.ndarray: Array of shape (3,) containing the next state
    """
    k1 = func(tn, xn)
    k2 = func((tn + h/2), (xn + np.multiply(k1,h/2)))
    k3 = func((tn + h/2), (xn + np.multiply(k2,h/2)))
    k4 = func((tn + h), (xn + np.multiply(k3, h)))
    k = np.round([k1, k2, k3, k4], 6).reshape(3,4)
    return xn + np.multiply(h/6,k[:,0] + np.multiply(2,k[:,1]) + np.multiply(2,k[:,2]) + k[:,3])


def helix(t, xi):
    """
    ODE function for helix

    Parameters:
        t (float): current time
        xi (list): current state

    Returns:
        list: List of derivatives [dx/dt, dy/dt, dz/dt]
    """
    w = 1
    k = 2
    x = -w*xi[1]
    y = w*xi[0]
    return [x, y, k]


def Lorenz(t, xi):
    """
    ODE function for Lorenz system

    Parameters:
        t (float): current time
        xi (list): current state

    Returns:
        list: List of derivatives [dx/dt, dy/dt, dz/dt]
    """
    sig = 10
    p = 28
    beta = 8/3
    x = sig*(xi[1]-xi[0])
    y = xi[0]*(p - xi[2])
    z = xi[0]*xi[1] - beta*xi[2]
    return [x, y, z]


def checkhelix(h, N, x0, tn_1):
    """
    True function for helix

    Parameters:
        h (float): step size
        N (int): number of steps
        x0 (list): initial state
        tn_1 (float): starting time

    Returns:
        numpy.ndarray: Array of shape (3, N+1) containing the true solution
    """
    x = np.zeros(shape=(3, N+1), dtype=np.float128)
    t = np.arange(tn_1, (N*h)+tn_1, h)
    x[:,0]=x0
    for i in range(0,N):
        x[:,i+1] = [3*np.cos(t[i]), 3*np.sin(t[i]), (2*t[i])]
    return x


def helix_demo():
    """
    Calculator with graph for helix (for testing only)
    """
    pred_hel = AB2_3D(func=helix, h=.1, N=500, x0=[3, 0, 0], x1=[1.62, 2.52, 2], tn_1=1)
    tru_hel = checkhelix(h=.1, N=500, x0=[3, 0, 0], tn_1=1)

    fig, ax = plt.subplots(1,2, subplot_kw={'projection': "3d"})
    ax[0].plot(pred_hel[0,:], pred_hel[1,:], pred_hel[2,:])
    ax[1].plot(tru_hel[0,:], tru_hel[1,:], tru_hel[2,:])
    plt.show()


def lorenz_calc():
    """
    Calculator with graph for Lorenz system (for testing only)
    """
    pred_lor = AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1,1,1], x1=RK4(Lorenz, h=.0025, xn=[1,1,1], tn=0), tn_1=0)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pred_lor[0,:], pred_lor[1,:], pred_lor[2,:], lw=.3)
    plt.show()
    

if __name__ == '__main__':
    pred_lor = [AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1,1,1], x1=RK4(Lorenz, h=.0025, xn=[1,1,1], tn=0), tn_1=0),
                AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1.01,1,1], x1=RK4(Lorenz, h=.0025, xn=[1.01,1,1], tn=0), tn_1=0),
                AB2_3D(func=Lorenz, h=.01, N=5000, x0=[1.3,1,1], x1=RK4(Lorenz, h=.0025, xn=[1.3,1,1], tn=0), tn_1=0)]

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
