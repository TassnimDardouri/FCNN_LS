import numpy as np
from skimage import io, transform
import skimage
from matplotlib import pyplot as plt
import pickle
import scipy.linalg as slin
import os
import scipy.io
from scipy import signal


def coef_P(x,y):
    Rupdx = np.matmul(np.transpose(x), x)
    rjourx = np.sum(np.multiply(x,y), axis = 0)
    rjourx = rjourx.reshape(rjourx.shape[0],1)
    if slin.det(Rupdx) == 0:
        p = np.matmul((slin.pinv(Rupdx)),rjourx)
    else:
        p = np.matmul((slin.inv(Rupdx)),rjourx)
    p = p.reshape(1,p.shape[0])
    pred = np.expand_dims(np.sum(np.multiply(x,p),axis=1),axis=1)
    return pred, p


def coef_U(image,x,h_2d):
    y_tild_ = signal.convolve2d(image, h_2d , boundary='symm', mode='same')
    y_tild_flat = np.expand_dims(y_tild_[::2,::2].flatten(),axis=1)
    x0_flat = np.expand_dims(image[::2,::2].flatten(),axis=1)
    Rupdx = np.matmul(np.transpose(x),x)
    r1 = np.multiply(x,y_tild_flat)
    r2 = np.multiply(x,x0_flat)
    rjourx = np.expand_dims(np.sum(r1-r2,axis=0),axis=1)
    if slin.det(Rupdx) == 0:
        p = np.matmul((slin.pinv(Rupdx)),rjourx)
    else:
        p = np.matmul((slin.inv(Rupdx)),rjourx)
    p = p.reshape(1,p.shape[0])
    pred = np.expand_dims(np.sum(np.multiply(x,p),axis=1),axis=1)
    return pred, p


def coef_fixe_P_h_v(x):
    p = np.expand_dims(np.array([0.5,0.5,-0.25,-0.25]),axis=0)
    pred = np.expand_dims(np.sum(np.multiply(x,p),axis=1),axis=1)
    return pred


def coef_fixe_P_d(x):
    p = np.expand_dims(np.array([-0.25,-0.25,-0.25,-0.25,0.5,0.5,0.5,0.5]),axis=0)
    pred = np.expand_dims(np.sum(np.multiply(x,p),axis=1),axis=1)
    return pred


def coef_fixe_U(x):
    p = np.expand_dims(np.array([0.25,0.25,0.25,0.25,-0.0625,-0.0625,-0.0625,-0.0625]),axis=0)
    pred = np.expand_dims(np.sum(np.multiply(x,p),axis=1),axis=1)
    return pred


#Compute YW prediction coefficients with for loop.
def coefs_opt_yule_walker_8ngh(image):
    x1 = []
    y1 = []
    a = pad_image_8ngh(image)
    nlig2, ncol2 = a.shape[0]//2, a.shape[1]//2
    Rupdx = np.zeros((8,8))
    rjourx = np.zeros((8,1))

    for i in range(nlig2):
        for j in range(ncol2):
            y = a[2*i+1,2*j+1]
            x = np.array([a[2*i,2*j],a[2*i,2*j+2],a[2*i+2,2*j],a[2*i+2,2*j+2],
                          a[2*i,2*j+1],a[2*i+2,2*j+1],
                          a[2*i+1,2*j],a[2*i+1,2*j+2]])

            y = np.double(y)
            x = np.double(x)
            x1.append(x)
            y1.append(y)
            x = x.reshape(x.shape[0],1)
            R = np.matmul(x, np.transpose(x))
            r = np.dot(y,x)

            Rupdx = Rupdx + R
            rjourx = rjourx + r
    if slin.det(Rupdx) == 0:
        p = np.matmul((slin.pinv(Rupdx)),rjourx)
    else:
        p = np.matmul((slin.inv(Rupdx)),rjourx)
    return x1, y1, p



def coef_U_forloop(image, xdh_, x_dv_, x_dd_):
    Rupdx_u = np.zeros((8,8))
    rjourx_u = np.zeros((8,1))
    # low pass filter
    N=15
    t = np.arange(-N,N+1)
    h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
    h_2d = h*np.transpose(h)
    y_tild_ = signal.convolve2d(image, h_2d , boundary='symm', mode='same')
    x0 = np.transpose(np.transpose(image[::2])[::2])
    # extract smoothed x0 from smoothed image
    y_tild = np.transpose(np.transpose(y_tild_[::2])[::2])
    x_dd = np.pad(x_dd_, ((1, 0),(1, 0)), "reflect")
    x_dh = np.pad(x_dh_, ((0, 0),(1, 0)), "reflect")
    x_dv = np.pad(x_dv_, ((1, 0),(0, 0)), "reflect")
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            xu = np.expand_dims(np.array([x_dh[i,j+1],x_dh[i,j],x_dv[i+1,j],x_dv[i,j],
                           x_dd[i+1,j+1],x_dd[i+1,j],x_dd[i,j+1],x_dd[i,j]]),axis=1)
            xu = np.double(xu)
            y_tild = np.double(y_tild)
            x0 = np.double(x0)
            R = np.matmul(xu,np.transpose(xu))
            r1 = xu*y_tild[i,j]
            r2 = xu*x0[i,j]
            r = r1 - r2
            Rupdx_u=Rupdx_u+R
            rjourx_u=rjourx_u+r

    if slin.det(Rupdx_u) == 0:
        p = np.matmul((slin.pinv(Rupdx_u)),rjourx_u)
    else:
        p = np.matmul((slin.inv(Rupdx_u)),rjourx_u)
    return x0, p
