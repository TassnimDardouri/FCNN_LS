import numpy as np
from scipy import signal
import sys

def reference_P3_42_AG_par(x0, x1, x2):

    x0_ = np.pad(x0, ((3, 2),(3, 2)), "symmetric")
    x0_ = np.delete(x0_,[1,2],0)
    x0_ = np.delete(x0_,[1,2],1)
    x1_ = np.pad(x1, ((3, 2),(0, 0)), "symmetric")
    x1_ = np.delete(x1_,[1,2],0)
    x1_ = np.pad(x1_, ((0,0),(1,1)), "reflect")
    x2_ = np.pad(x2, ((0, 0),(3, 2)), "symmetric")
    x2_ = np.delete(x2_,[1,2],1)
    x2_ = np.pad(x2_, ((1,1),(0,0)), "reflect")
    
    x0_1 = np.expand_dims(x0_[0:-3,0:-3].flatten(), axis = 1)
    x0_2 = np.expand_dims(x0_[0:-3,1:-2].flatten(), axis = 1)
    x0_3 = np.expand_dims(x0_[0:-3,2:-1].flatten(), axis = 1)
    x0_4 = np.expand_dims(x0_[0:-3,3:].flatten(), axis = 1)
    x0_5 = np.expand_dims(x0_[1:-2,0:-3].flatten(), axis = 1)
    x0_6 = np.expand_dims(x0_[1:-2,1:-2].flatten(), axis = 1)
    x0_7 = np.expand_dims(x0_[1:-2,2:-1].flatten(), axis = 1)
    x0_8 = np.expand_dims(x0_[1:-2,3:].flatten(), axis = 1)
    x0_9 = np.expand_dims(x0_[2:-1,0:-3].flatten(), axis = 1)
    x0_10 = np.expand_dims(x0_[2:-1,1:-2].flatten(), axis = 1)
    x0_11 = np.expand_dims(x0_[2:-1,2:-1].flatten(), axis = 1)
    x0_12 = np.expand_dims(x0_[2:-1,3:].flatten(), axis = 1)
    x0_13 = np.expand_dims(x0_[3:,0:-3].flatten(), axis = 1)
    x0_14 = np.expand_dims(x0_[3:,1:-2].flatten(), axis = 1)
    x0_15 = np.expand_dims(x0_[3:,2:-1].flatten(), axis = 1)
    x0_16 = np.expand_dims(x0_[3:,3:].flatten(), axis = 1)
    
    x1_1 = np.expand_dims(x1_[0:-3,0:-2].flatten(), axis = 1)
    x1_2 = np.expand_dims(x1_[1:-2,0:-2].flatten(), axis = 1)
    x1_3 = np.expand_dims(x1_[2:-1,0:-2].flatten(), axis = 1)
    x1_4 = np.expand_dims(x1_[3:,0:-2].flatten(), axis = 1)
    x1_5 = np.expand_dims(x1_[0:-3,1:-1].flatten(), axis = 1)
    x1_6 = np.expand_dims(x1_[1:-2,1:-1].flatten(), axis = 1)
    x1_7 = np.expand_dims(x1_[2:-1,1:-1].flatten(), axis = 1)
    x1_8 = np.expand_dims(x1_[3:,1:-1].flatten(), axis = 1)
    x1_9 = np.expand_dims(x1_[0:-3,2:].flatten(), axis = 1)
    x1_10 = np.expand_dims(x1_[1:-2,2:].flatten(), axis = 1)
    x1_11 = np.expand_dims(x1_[2:-1,2:].flatten(), axis = 1)
    x1_12 = np.expand_dims(x1_[3:,2:].flatten(), axis = 1)
    
    x2_1 = np.expand_dims(x2_[0:-2,0:-3].flatten(), axis = 1)
    x2_2 = np.expand_dims(x2_[0:-2,1:-2].flatten(), axis = 1)
    x2_3 = np.expand_dims(x2_[0:-2,2:-1].flatten(), axis = 1)
    x2_4 = np.expand_dims(x2_[0:-2,3:].flatten(), axis = 1)
    x2_5 = np.expand_dims(x2_[1:-1,0:-3].flatten(), axis = 1)
    x2_6 = np.expand_dims(x2_[1:-1,1:-2].flatten(), axis = 1)
    x2_7 = np.expand_dims(x2_[1:-1,2:-1].flatten(), axis = 1)
    x2_8 = np.expand_dims(x2_[1:-1,3:].flatten(), axis = 1)
    x2_9 = np.expand_dims(x2_[2:,0:-3].flatten(), axis = 1)
    x2_10 = np.expand_dims(x2_[2:,1:-2].flatten(), axis = 1)
    x2_11 = np.expand_dims(x2_[2:,2:-1].flatten(), axis = 1)
    x2_12 = np.expand_dims(x2_[2:,3:].flatten(), axis = 1)
    
    ref_P3 = np.concatenate((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x0_13, x0_14, x0_15, x0_16, x1_1, 
                        x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8, x1_9, x1_10,
                        x1_11, x1_12, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, 
                        x2_8, x2_9, x2_10, x2_11, x2_12), axis = 1)
    return ref_P3

def reference_P2_42_2D_AG_par(x0, x1, x_dd):
    
    x02 = np.pad(x0, ((3, 2),(0, 0)), "symmetric")
    x02 = np.delete(x02,[1,2],0)
    x02 = np.pad(x02, ((0, 0),(1, 1)), "reflect")
    
    x1_ = np.pad(x1, ((0,1),(2,3)), "symmetric")
    H,W = x1_.shape
    x1_ = np.delete(x1_, [W-2,W-3],1)
    
    x_dd2 = np.pad(x_dd, ((0, 0),(1, 0)), "edge")
    
    x0_1 = np.expand_dims(x02[0:-3,0:-2].flatten(), axis = 1)
    x0_2 = np.expand_dims(x02[0:-3,1:-1].flatten(), axis = 1)
    x0_3 = np.expand_dims(x02[0:-3,2:].flatten(), axis = 1)
    x0_4 = np.expand_dims(x02[1:-2,0:-2].flatten(), axis = 1)
    x0_5 = np.expand_dims(x02[1:-2,1:-1].flatten(), axis = 1)
    x0_6 = np.expand_dims(x02[1:-2,2:].flatten(), axis = 1)
    x0_7 = np.expand_dims(x02[2:-1,0:-2].flatten(), axis = 1)
    x0_8 = np.expand_dims(x02[2:-1,1:-1].flatten(), axis = 1)
    x0_9 = np.expand_dims(x02[2:-1,2:].flatten(), axis = 1)
    x0_10 = np.expand_dims(x02[3:,0:-2].flatten(), axis = 1)
    x0_11 = np.expand_dims(x02[3:,1:-1].flatten(), axis = 1)
    x0_12 = np.expand_dims(x02[3:,2:].flatten(), axis = 1)
    
    x1_1 = np.expand_dims(x1_[0:-1,0:-3].flatten(), axis = 1)
    x1_2 = np.expand_dims(x1_[0:-1,1:-2].flatten(), axis = 1)
    x1_3 = np.expand_dims(x1_[0:-1,2:-1].flatten(), axis = 1)
    x1_4 = np.expand_dims(x1_[0:-1,3:].flatten(), axis = 1)
    x1_5 = np.expand_dims(x1_[1:,0:-3].flatten(), axis = 1)
    x1_6 = np.expand_dims(x1_[1:,1:-2].flatten(), axis = 1)
    x1_7 = np.expand_dims(x1_[1:,2:-1].flatten(), axis = 1)
    x1_8 = np.expand_dims(x1_[1:,3:].flatten(), axis = 1)
    
    x_dd2_1 = np.expand_dims(x_dd2[:,0:-1].flatten(), axis = 1)
    x_dd2_2 = np.expand_dims(x_dd2[:,1:].flatten(), axis = 1)
    
    ref_P2 = np.concatenate((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, 
                        x1_7, x1_8, x_dd2_2, x_dd2_1), axis = 1)
    
    return ref_P2

def reference_P1_42_2D_AG_par(x0, x_dd):
    
    x01 = np.pad(x0, ((0, 0),(3, 2)), "symmetric")
    x01 = np.delete(x01,[1,2],1)
    x01 = np.pad(x01, ((1, 1),(0, 0)), "reflect")
    
    x_dd1 = np.pad(x_dd, ((1, 0),(0, 0)), "edge")
    
    x0_1 = np.expand_dims(x01[0:-2,0:-3].flatten(), axis = 1)
    x0_2 = np.expand_dims(x01[0:-2,1:-2].flatten(), axis = 1)
    x0_3 = np.expand_dims(x01[0:-2,2:-1].flatten(), axis = 1)
    x0_4 = np.expand_dims(x01[0:-2,3:].flatten(), axis = 1)
    x0_5 = np.expand_dims(x01[1:-1,0:-3].flatten(), axis = 1)
    x0_6 = np.expand_dims(x01[1:-1,1:-2].flatten(), axis = 1)
    x0_7 = np.expand_dims(x01[1:-1,2:-1].flatten(), axis = 1)
    x0_8 = np.expand_dims(x01[1:-1,3:].flatten(), axis = 1)
    x0_9 = np.expand_dims(x01[2:,0:-3].flatten(), axis = 1)
    x0_10 = np.expand_dims(x01[2:,1:-2].flatten(), axis = 1)
    x0_11 = np.expand_dims(x01[2:,2:-1].flatten(), axis = 1)
    x0_12 = np.expand_dims(x01[2:,3:].flatten(), axis = 1)
    
    x_dd1_1 = np.expand_dims(x_dd1[0:-1,:].flatten(), axis = 1)
    x_dd1_2 = np.expand_dims(x_dd1[1:,:].flatten(), axis = 1)
    
    ref_P1 = np.concatenate((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x_dd1_2, x_dd1_1),1)
    return ref_P1

def target_P3_par(x3):
    
    y_P3 = np.expand_dims(x3.flatten(), axis = 1)
    return y_P3

def target_P1_P2_par(x1,x2):
    
    y_P1 = np.expand_dims(x1.flatten(), axis = 1)
    y_P2 = np.expand_dims(x2.flatten(), axis = 1)
    return y_P1, y_P2

def target_U_par(image, x0):
               
    N = 15
    t = np.arange(-N,N+1)
    h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
    h_2d = h*np.transpose(h)
    y_tild_ = signal.convolve2d(image, h_2d , boundary='symm', mode='same')
    y_tild = y_tild_[::2,::2]
    
    y_U = np.expand_dims(y_tild.flatten(), axis = 1)-np.expand_dims(x0.flatten(), axis = 1)
    
    return y_U

def reference_U_par(x_dd, x_dv, x_dh):
    
    x_dd1 = np.pad(x_dd, ((1, 0),(1, 0)), "edge")
    x_dh1 = np.pad(x_dh, ((0, 0),(1, 0)), "edge")
    x_dv1 = np.pad(x_dv, ((1, 0),(0, 0)), "edge")
    
    x_dd_1 = np.expand_dims(x_dd1[:-1,:-1].flatten(), axis = 1)
    x_dd_2 = np.expand_dims(x_dd1[:-1,1:].flatten(), axis = 1)
    x_dd_3 = np.expand_dims(x_dd1[1:,:-1].flatten(), axis = 1)
    x_dd_4 = np.expand_dims(x_dd1[1:,1:].flatten(), axis = 1)
    
    x_dh_1 = np.expand_dims(x_dh1[0:,0:-1].flatten(), axis = 1)
    x_dh_2 = np.expand_dims(x_dh1[0:,1:].flatten(), axis = 1)
    
    x_dv_1 = np.expand_dims(x_dv1[0:-1,0:].flatten(), axis = 1)
    x_dv_2 = np.expand_dims(x_dv1[1:,:].flatten(), axis = 1)
 
    ref_U = np.concatenate((x_dh_2, x_dh_1, x_dv_2, x_dv_1, x_dd_4, x_dd_3, x_dd_2, x_dd_1), axis = 1)
    
    return ref_U