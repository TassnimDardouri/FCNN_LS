import numpy as np
from scipy import signal

def Ref_P3(trans, x0, x1, x2):
    if trans == '22':
        ref_P3 = reference_P3(x0, x1, x2)
    elif trans in ['42', '42_2D']:
        ref_P3 = reference_P3_42(x0, x1, x2)
    elif trans == '42_AG':
        ref_P3 = reference_P3_42_AG(x0, x1, x2)
    elif trans == '62_2D':
        ref_P3 = reference_P3_62_2D(x0, x1, x2)
    else:
        raise ValueError('please specify reference type')
    return ref_P3
        
def Ref_P1_P2(trans, x0, x_dd):
    if trans =='22':
        ref_P1, ref_P2 = reference_P1_P2(x0, x_dd)
    elif trans == '42':
        ref_P1, ref_P2 = reference_P1_P2_42(x0, x_dd)
    elif trans == '42_2D':
        ref_P1, ref_P2 = reference_P1_P2_42_2D(x0, x_dd)
    elif trans == '62_2D':
        ref_P1, ref_P2 = reference_P1_P2_62_2D(x0, x_dd)
    else:
        raise ValueError('please specify reference type')
    return ref_P1, ref_P2
    
    
def target_U(image, x0):
    y_U = np.zeros((x0.shape[0]*x0.shape[1],1))
    N = 15
    t = np.arange(-N,N+1)
    h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
    h_2d = h*np.transpose(h)
    y_tild_ = signal.convolve2d(image, h_2d , boundary='symm', mode='same')
    y_tild = y_tild_[::2,::2]
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            y_U[n,:] = y_tild[i,j]-x0[i,j]
            n = n+1
    return y_U


def reference_U(x_dd, x_dv, x_dh):
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_U = np.zeros((num_pixels,8))
    x_dd1 = np.pad(x_dd, ((1, 0),(1, 0)), "edge")
    x_dh1 = np.pad(x_dh, ((0, 0),(1, 0)), "edge")
    x_dv1 = np.pad(x_dv, ((1, 0),(0, 0)), "edge")

    
    n=0
    for i in range(x_dd.shape[0]):
        for j in range(x_dd.shape[1]):
            ref_U[n,0] = x_dh1[i,j+1]
            ref_U[n,1] = x_dh1[i,j]
            ref_U[n,2] = x_dv1[i+1,j]
            ref_U[n,3] = x_dv1[i,j]
            ref_U[n,4] = x_dd1[i+1,j+1]
            ref_U[n,5] = x_dd1[i+1,j]
            ref_U[n,6] = x_dd1[i,j+1]
            ref_U[n,7] = x_dd1[i,j]
            n = n+1
    return ref_U



def target_P1_P2(x1,x2):
    n=0
    num_pixels = x1.shape[0]*x1.shape[1]
    y_P1 = np.zeros((num_pixels,1))
    y_P2 = np.zeros((num_pixels,1))
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            y_P1[n,:] = x1[i,j]
            y_P2[n,:] = x2[i,j]
            n = n+1
    return y_P1, y_P2



def reference_P1_P2(x0, x_dd):
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_P1 = np.zeros((num_pixels,4))
    ref_P2 = np.zeros((num_pixels,4))
    
    x01 = np.pad(x0, ((0, 0),(0, 1)), "edge")
    x02 = np.pad(x0, ((0, 1),(0, 0)), "edge")
    x_dd1 = np.pad(x_dd, ((1, 0),(0, 0)), "edge")
    x_dd2 = np.pad(x_dd, ((0, 0),(1, 0)), "edge")
    
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            ref_P1[n,0] = x01[i,j]
            ref_P1[n,1] = x01[i,j+1]
            ref_P1[n,2] = x_dd1[i+1,j]
            ref_P1[n,3] = x_dd1[i,j]
            ref_P2[n,0] = x02[i,j]
            ref_P2[n,1] = x02[i+1,j]
            ref_P2[n,2] = x_dd2[i,j+1]
            ref_P2[n,3] = x_dd2[i,j]
            n = n+1
    return ref_P1, ref_P2



def target_P3(x3):
    n=0
    y_P3 = np.zeros((x3.shape[0]*x3.shape[1],1))
    for i in range(x3.shape[0]):
        for j in range(x3.shape[1]):
            y_P3[n,:] = x3[i,j]
            n = n+1
    return y_P3



def reference_P3(x0, x1, x2):
    num_pixels = x0.shape[0]*x0.shape[1]
    ref_P3 = np.zeros((num_pixels,8))
    x0_ = np.pad(x0, ((0, 1),(0, 1)), "edge")
    x1_ = np.pad(x1, ((0, 1),(0, 0)), "edge")
    x2_ = np.pad(x2, ((0, 0),(0, 1)), "edge")
    
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            ref_P3[n,0] = x0_[i,j]
            ref_P3[n,1] = x0_[i,j+1]
            ref_P3[n,2] = x0_[i+1,j]
            ref_P3[n,3] = x0_[i+1,j+1]
            ref_P3[n,4] = x1_[i,j]
            ref_P3[n,5] = x1_[i+1,j]
            ref_P3[n,6] = x2_[i,j]
            ref_P3[n,7] = x2_[i,j+1]
            n = n+1
    return ref_P3

def reference_P3_42(x0, x1, x2):
    num_pixels = x0.shape[0]*x0.shape[1]
    ref_P3 = np.zeros((num_pixels,24))
    x0_ = np.pad(x0, ((3, 2),(3, 2)), "symmetric")
    x0_ = np.delete(x0_,[1,2],0)
    x0_ = np.delete(x0_,[1,2],1)
    x1_ = np.pad(x1, ((3, 2),(0, 0)), "symmetric")
    x1_ = np.delete(x1_,[1,2],0)
    x2_ = np.pad(x2, ((0, 0),(3, 2)), "symmetric")
    x2_ = np.delete(x2_,[1,2],1)
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            ref_P3[n,0] = x0_[i,j]
            ref_P3[n,1] = x0_[i,j+1]
            ref_P3[n,2] = x0_[i,j+2]
            ref_P3[n,3] = x0_[i,j+3]
            
            ref_P3[n,4] = x0_[i+1,j]
            ref_P3[n,5] = x0_[i+1,j+1]
            ref_P3[n,6] = x0_[i+1,j+2]
            ref_P3[n,7] = x0_[i+1,j+3]
            
            ref_P3[n,8] = x0_[i+2,j]
            ref_P3[n,9] = x0_[i+2,j+1]
            ref_P3[n,10] = x0_[i+2,j+2]
            ref_P3[n,11] = x0_[i+2,j+3]
            
            ref_P3[n,12] = x0_[i+3,j]
            ref_P3[n,13] = x0_[i+3,j+1]
            ref_P3[n,14] = x0_[i+3,j+2]
            ref_P3[n,15] = x0_[i+3,j+3]
            
            ref_P3[n,16] = x1_[i,j]
            ref_P3[n,17] = x1_[i+1,j]
            ref_P3[n,18] = x1_[i+2,j]
            ref_P3[n,19] = x1_[i+3,j]
            
            ref_P3[n,20] = x2_[i,j]
            ref_P3[n,21] = x2_[i,j+1]
            ref_P3[n,22] = x2_[i,j+2]
            ref_P3[n,23] = x2_[i,j+3]
            n = n+1  
    return ref_P3

def reference_P3_42_AG(x0, x1, x2):
    num_pixels = x0.shape[0]*x0.shape[1]
    ref_P3 = np.zeros((num_pixels,40))
    x0_ = np.pad(x0, ((3, 2),(3, 2)), "symmetric")
    x0_ = np.delete(x0_,[1,2],0)
    x0_ = np.delete(x0_,[1,2],1)
    x1_ = np.pad(x1, ((3, 2),(0, 0)), "symmetric")
    x1_ = np.delete(x1_,[1,2],0)
    x1_ = np.pad(x1_, ((0,0),(1,1)), "reflect")
    x2_ = np.pad(x2, ((0, 0),(3, 2)), "symmetric")
    x2_ = np.delete(x2_,[1,2],1)
    x2_ = np.pad(x2_, ((1,1),(0,0)), "reflect")
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            c=0
            for l in range(4):
                ref_P3[n,c+0] = x0_[i+l,j]
                ref_P3[n,c+1] = x0_[i+l,j+1]
                ref_P3[n,c+2] = x0_[i+l,j+2]
                ref_P3[n,c+3] = x0_[i+l,j+3]
                c+=4
            
            for l in range(3):

                ref_P3[n,c+0] = x1_[i,j+l]
                ref_P3[n,c+1] = x1_[i+1,j+l]
                ref_P3[n,c+2] = x1_[i+2,j+l]
                ref_P3[n,c+3] = x1_[i+3,j+l]

                c+=4
            for l in range(3):

                ref_P3[n,c+0] = x2_[i+l,j]
                ref_P3[n,c+1] = x2_[i+l,j+1]
                ref_P3[n,c+2] = x2_[i+l,j+2]
                ref_P3[n,c+3] = x2_[i+l,j+3]

                c+=4

            n = n+1  
    return ref_P3

def reference_P3_42_AG_new(x0, x1, x2):
    num_pixels = x0.shape[0]*x0.shape[1]
    ref_P3 = np.zeros((num_pixels,40))
    x0_ = np.pad(x0, ((3, 2),(3, 2)), "symmetric")
    x0_ = np.delete(x0_,[1,2],0)
    x0_ = np.delete(x0_,[1,2],1)
    x1_ = np.pad(x1, ((3, 2),(0, 0)), "symmetric")
    x1_ = np.delete(x1_,[1,2],0)
    x1_ = np.pad(x1_, ((0,0),(1,1)), "reflect")
    x2_ = np.pad(x2, ((0, 0),(3, 2)), "symmetric")
    x2_ = np.delete(x2_,[1,2],1)
    x2_ = np.pad(x2_, ((1,1),(0,0)), "reflect")
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            c=0
            for l in range(4):
                ref_P3[n,c+0] = x0_[i+l,j]
                ref_P3[n,c+1] = x0_[i+l,j+1]
                ref_P3[n,c+2] = x0_[i+l,j+2]
                ref_P3[n,c+3] = x0_[i+l,j+3]

                c+=4
            for l in range(4):

                ref_P3[n,c+0] = x1_[i+l,j+0]
                ref_P3[n,c+1] = x1_[i+l,j+1]
                ref_P3[n,c+2] = x1_[i+l,j+2]

                c+=3
            
            for l in range(3):

                ref_P3[n,c+0] = x2_[i+l,j]
                ref_P3[n,c+1] = x2_[i+l,j+1]
                ref_P3[n,c+2] = x2_[i+l,j+2]
                ref_P3[n,c+3] = x2_[i+l,j+3]

                c+=4

            n = n+1  
    return ref_P3

def reference_P3_62_2D(x0, x1, x2):
    
    num_pixels = x0.shape[0]*x0.shape[1]
    ref_P3 = np.zeros((num_pixels,48))
    
    x0_ = np.pad(x0, ((4, 3),(4, 3)), "symmetric")
    x0_ = np.delete(x0_,[2,3],0)
    x0_ = np.delete(x0_,[2,3],1)
    
    x1_ = np.pad(x1, ((4, 3),(0, 0)), "symmetric")
    x1_ = np.delete(x1_,[2,3],0)
    
    x2_ = np.pad(x2, ((0, 0), (4, 3)), "symmetric")
    x2_ = np.delete(x2_,[2,3],1)
    
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            count = 0
            
            for l in range(6):
                ref_P3[n,count] = x0_[i+l,j]
                ref_P3[n,count+1] = x0_[i+l,j+1]
                ref_P3[n,count+2] = x0_[i+l,j+2]
                ref_P3[n,count+3] = x0_[i+l,j+3]
                ref_P3[n,count+4] = x0_[i+l,j+4]
                ref_P3[n,count+5] = x0_[i+l,j+5]
                count += 6 
                
            for l in range(6):
                ref_P3[n,count] = x1_[i+l,j]
                count += 1
                
            ref_P3[n,count] = x2_[i,j]
            ref_P3[n,count+1] = x2_[i,j+1]
            ref_P3[n,count+2] = x2_[i,j+2]
            ref_P3[n,count+3] = x2_[i,j+3]
            ref_P3[n,count+4] = x2_[i,j+4]
            ref_P3[n,count+5] = x2_[i,j+5]
            count += 6
                
            n = n+1
            
    return ref_P3.astype('float64')

def reference_P1_P2_42(x0, x_dd):
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_P1 = np.zeros((num_pixels,6))
    ref_P2 = np.zeros((num_pixels,6))
    
    x01 = np.pad(x0, ((0, 0),(3, 2)), "symmetric")
    x01 = np.delete(x01,[1,2],1)
    x02 = np.pad(x0, ((3, 2),(0, 0)), "symmetric")
    x02 = np.delete(x02,[1,2],0)
    x_dd1 = np.pad(x_dd, ((1, 0),(0, 0)), "edge")
    x_dd2 = np.pad(x_dd, ((0, 0),(1, 0)), "edge")
    
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            
            ref_P1[n,0] = x01[i,j]
            ref_P1[n,1] = x01[i,j+1]
            ref_P1[n,2] = x01[i,j+2]
            ref_P1[n,3] = x01[i,j+3]
            ref_P1[n,4] = x_dd1[i+1,j]
            ref_P1[n,5] = x_dd1[i,j]
            
            ref_P2[n,0] = x02[i,j]
            ref_P2[n,1] = x02[i+1,j]
            ref_P2[n,2] = x02[i+2,j]
            ref_P2[n,3] = x02[i+3,j]
            ref_P2[n,4] = x_dd2[i,j+1]
            ref_P2[n,5] = x_dd2[i,j]
            n = n+1
            
    return ref_P1, ref_P2

def reference_P1_P2_42_2D(x0, x_dd):
    
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_P1 = np.zeros((num_pixels,14))
    ref_P2 = np.zeros((num_pixels,14))
    
    x01 = np.pad(x0, ((0, 0),(3, 2)), "symmetric")
    x01 = np.delete(x01,[1,2],1)
    x01 = np.pad(x01, ((1, 1),(0, 0)), "reflect")
    x02 = np.pad(x0, ((3, 2),(0, 0)), "symmetric")
    x02 = np.delete(x02,[1,2],0)
    x02 = np.pad(x02, ((0, 0),(1, 1)), "reflect")
    x_dd1 = np.pad(x_dd, ((1, 0),(0, 0)), "edge")
    x_dd2 = np.pad(x_dd, ((0, 0),(1, 0)), "edge")
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            
            ref_P1[n,0] = x01[i,j] 
            ref_P1[n,1] = x01[i,j+1]
            ref_P1[n,2] = x01[i,j+2]
            ref_P1[n,3] = x01[i,j+3]
            
            ref_P1[n,4] = x01[i+1,j]
            ref_P1[n,5] = x01[i+1,j+1]
            ref_P1[n,6] = x01[i+1,j+2]
            ref_P1[n,7] = x01[i+1,j+3]
            
            ref_P1[n,8] = x01[i+2,j]
            ref_P1[n,9] = x01[i+2,j+1]
            ref_P1[n,10] = x01[i+2,j+2]
            ref_P1[n,11] = x01[i+2,j+3]
            
            ref_P1[n,12] = x_dd1[i+1,j]
            ref_P1[n,13] = x_dd1[i,j]
            
            
            ref_P2[n,0] = x02[i,j]
            ref_P2[n,1] = x02[i,j+1]
            ref_P2[n,2] = x02[i,j+2]
            
            ref_P2[n,3] = x02[i+1,j]
            ref_P2[n,4] = x02[i+1,j+1]
            ref_P2[n,5] = x02[i+1,j+2]
            
            ref_P2[n,6] = x02[i+2,j]
            ref_P2[n,7] = x02[i+2,j+1]
            ref_P2[n,8] = x02[i+2,j+2]
            
            ref_P2[n,9] = x02[i+3,j]
            ref_P2[n,10] = x02[i+3,j+1]
            ref_P2[n,11] = x02[i+3,j+2]
            
            ref_P2[n,12] = x_dd2[i,j+1]
            ref_P2[n,13] = x_dd2[i,j]
            n = n+1
            
    return ref_P1, ref_P2

def reference_P2_42_2D_AG(x0, x1, x_dd):
    
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_P2 = np.zeros((num_pixels,22))
    
    x02 = np.pad(x0, ((3, 2),(0, 0)), "symmetric")
    x02 = np.delete(x02,[1,2],0)
    x02 = np.pad(x02, ((0, 0),(1, 1)), "reflect")
    
    x1_ = np.pad(x1, ((0,1),(2,3)), "symmetric")
    H,W = x1_.shape
    x1_ = np.delete(x1_, [W-2,W-3],1)
    
    x_dd2 = np.pad(x_dd, ((0, 0),(1, 0)), "edge")
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            
            ref_P2[n,0] = x02[i,j]
            ref_P2[n,1] = x02[i,j+1]
            ref_P2[n,2] = x02[i,j+2]
            
            ref_P2[n,3] = x02[i+1,j]
            ref_P2[n,4] = x02[i+1,j+1]
            ref_P2[n,5] = x02[i+1,j+2]
            
            ref_P2[n,6] = x02[i+2,j]
            ref_P2[n,7] = x02[i+2,j+1]
            ref_P2[n,8] = x02[i+2,j+2]
            
            ref_P2[n,9] = x02[i+3,j]
            ref_P2[n,10] = x02[i+3,j+1]
            ref_P2[n,11] = x02[i+3,j+2]
            
            ref_P2[n,12] = x1_[i,j]
            ref_P2[n,13] = x1_[i,j+1]
            ref_P2[n,14] = x1_[i,j+2]
            ref_P2[n,15] = x1_[i,j+3]
            
            ref_P2[n,16] = x1_[i+1,j]
            ref_P2[n,17] = x1_[i+1,j+1]
            ref_P2[n,18] = x1_[i+1,j+2]
            ref_P2[n,19] = x1_[i+1,j+3]
            
            ref_P2[n,20] = x_dd2[i,j+1]
            ref_P2[n,21] = x_dd2[i,j]
            n = n+1
            
    return ref_P2

def reference_P1_42_2D_AG(x0, x_dd):
    
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_P1 = np.zeros((num_pixels,14))
    
    x01 = np.pad(x0, ((0, 0),(3, 2)), "symmetric")
    x01 = np.delete(x01,[1,2],1)
    x01 = np.pad(x01, ((1, 1),(0, 0)), "reflect")
    
    x_dd1 = np.pad(x_dd, ((1, 0),(0, 0)), "edge")
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            
            ref_P1[n,0] = x01[i,j] 
            ref_P1[n,1] = x01[i,j+1]
            ref_P1[n,2] = x01[i,j+2]
            ref_P1[n,3] = x01[i,j+3]
            
            ref_P1[n,4] = x01[i+1,j]
            ref_P1[n,5] = x01[i+1,j+1]
            ref_P1[n,6] = x01[i+1,j+2]
            ref_P1[n,7] = x01[i+1,j+3]
            
            ref_P1[n,8] = x01[i+2,j]
            ref_P1[n,9] = x01[i+2,j+1]
            ref_P1[n,10] = x01[i+2,j+2]
            ref_P1[n,11] = x01[i+2,j+3]
            
            ref_P1[n,12] = x_dd1[i+1,j]
            ref_P1[n,13] = x_dd1[i,j]
            n = n+1
            
    return ref_P1

def reference_P1_P2_62_2D(x0, x_dd):
    
    num_pixels = x_dd.shape[0]*x_dd.shape[1]
    ref_P1 = np.zeros((num_pixels,20))
    ref_P2 = np.zeros((num_pixels,20))
    
    x01 = np.pad(x0, ((0, 0),(4, 3)), "symmetric")
    x01 = np.delete(x01,[2,3],1)
    x01 = np.pad(x01, ((1, 1),(0, 0)), "reflect")
    x02 = np.pad(x0, ((4, 3),(0, 0)), "symmetric")
    x02 = np.delete(x02,[2,3],0)
    x02 = np.pad(x02, ((0, 0),(1, 1)), "reflect")
    x_dd1 = np.pad(x_dd, ((1, 0),(0, 0)), "edge")
    x_dd2 = np.pad(x_dd, ((0, 0),(1, 0)), "edge")
    
    n = 0
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            
            ref_P1[n,0] = x01[i,j] 
            ref_P1[n,1] = x01[i,j+1]
            ref_P1[n,2] = x01[i,j+2]
            ref_P1[n,3] = x01[i,j+3]
            ref_P1[n,4] = x01[i,j+4]
            ref_P1[n,5] = x01[i,j+5]
            
            ref_P1[n,6] = x01[i+1,j]
            ref_P1[n,7] = x01[i+1,j+1]
            ref_P1[n,8] = x01[i+1,j+2]
            ref_P1[n,9] = x01[i+1,j+3]
            ref_P1[n,10] = x01[i+1,j+4]
            ref_P1[n,11] = x01[i+1,j+5]
            
            ref_P1[n,12] = x01[i+2,j]
            ref_P1[n,13] = x01[i+2,j+1]
            ref_P1[n,14] = x01[i+2,j+2]
            ref_P1[n,15] = x01[i+2,j+3]
            ref_P1[n,16] = x01[i+2,j+4]
            ref_P1[n,17] = x01[i+2,j+5]
            
            ref_P1[n,18] = x_dd1[i+1,j]
            ref_P1[n,19] = x_dd1[i,j]
            
            
            ref_P2[n,0] = x02[i,j]
            ref_P2[n,1] = x02[i,j+1]
            ref_P2[n,2] = x02[i,j+2]
            
            ref_P2[n,3] = x02[i+1,j]
            ref_P2[n,4] = x02[i+1,j+1]
            ref_P2[n,5] = x02[i+1,j+2]
            
            ref_P2[n,6] = x02[i+2,j]
            ref_P2[n,7] = x02[i+2,j+1]
            ref_P2[n,8] = x02[i+2,j+2]
            
            ref_P2[n,9] = x02[i+3,j]
            ref_P2[n,10] = x02[i+3,j+1]
            ref_P2[n,11] = x02[i+3,j+2]
            
            ref_P2[n,12] = x02[i+4,j]
            ref_P2[n,13] = x02[i+4,j+1]
            ref_P2[n,14] = x02[i+4,j+2]
            
            ref_P2[n,15] = x02[i+5,j]
            ref_P2[n,16] = x02[i+5,j+1]
            ref_P2[n,17] = x02[i+5,j+2]
            
            ref_P2[n,18] = x_dd2[i,j+1]
            ref_P2[n,19] = x_dd2[i,j]
            n = n+1
            
    return ref_P1.astype('float64'), ref_P2.astype('float64')