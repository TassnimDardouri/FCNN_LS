import numpy as np
import skimage.metrics
import math
import sys
from scipy.io import loadmat
from matplotlib import pyplot as plt

sys.path.append('/data/tdardour_data/image_comp/codes')
from utils.data_utils import subband
from utils.linear_utils import (coef_fixe_P_d, coef_fixe_P_h_v, 
                                coef_fixe_U, coef_P, coef_U)
from utils.references import (Ref_P3, target_P3, 
                              Ref_P1_P2, target_P1_P2, 
                              reference_P1_42_2D_AG, reference_P2_42_2D_AG,
                              reference_U, target_U)
from utils.references_par import (reference_P3_42_AG_par,
                                 reference_P2_42_2D_AG_par, 
                                 reference_P1_42_2D_AG_par,
                                 reference_U_par,
                                 target_P1_P2_par, 
                                 target_P3_par,
                                 target_U_par)

from utils.model_utils import adaptive_predict, custom_loss

def Forward(img, dynamic, lossy, trans, model_p1, model_p2, model_p3, model_u):
    
    if lossy == False:
        img = np.round(img)
        
    x0, x1, x2, x3 = subband(img)
    num_pixels = x0.shape[0]*x0.shape[1]
    print(img.shape)
    
    # generate reference vector for P(HH)
    ref_P3 = Ref_P3(trans, x0, x1, x2)
    y_P3 = target_P3(x3)
    if model_p3 == 0:

        if dynamic == 0:
            
            diag_pred = coef_fixe_P_d(ref_P3)
            if lossy:
                x_dd = (y_P3 - diag_pred).reshape(x0.shape[0],x0.shape[1])
            else:
                x_dd = (y_P3 - np.round(diag_pred)).reshape(x0.shape[0],x0.shape[1])
                
            if trans == '42_AG':
                ref_P1 = reference_P1_42_2D_AG(x0, x_dd)
                ref_P2 = reference_P2_42_2D_AG(x0, x1, x_dd)
            else:
                ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
                
            y_P1, y_P2 = target_P1_P2(x1,x2)
            vertic_pred = coef_fixe_P_h_v(ref_P2)
            horiz_pred = coef_fixe_P_h_v(ref_P1)
            
            if lossy:
                
                x_dv = (y_P2 - vertic_pred).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - horiz_pred).reshape(x0.shape[0],x0.shape[1])
                
            else:
                
                x_dv = (y_P2 - np.round(vertic_pred)).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - np.round(horiz_pred)).reshape(x0.shape[0],x0.shape[1])
                
            ref_U = reference_U(x_dd, x_dv, x_dh)
            y_U = target_U(img, x0)
            im_pred_flat = coef_fixe_U(ref_U)
            
        else:
            
            diag_pred, p3 = coef_P(ref_P3,y_P3)
            
            if lossy:
                x_dd = (y_P3 - diag_pred).reshape(x0.shape[0],x0.shape[1])
            else:
                x_dd = (y_P3 - np.round(diag_pred)).reshape(x0.shape[0],x0.shape[1])
                
            if trans == '42_AG':
                ref_P1 = reference_P1_42_2D_AG(x0, x_dd)
                ref_P2 = reference_P2_42_2D_AG(x0, x1, x_dd)
                print('ref P1 P2 42_AG')
            else:
                ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
                
            y_P1, y_P2 = target_P1_P2(x1,x2)
            vertic_pred, p2 = coef_P(ref_P2,y_P2)
            horiz_pred, p1 = coef_P(ref_P1,y_P1)
            
            if lossy:
                
                x_dv = (y_P2 - vertic_pred).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - horiz_pred).reshape(x0.shape[0],x0.shape[1])
                
            else:
                
                x_dv = (y_P2 - np.round(vertic_pred)).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - np.round(horiz_pred)).reshape(x0.shape[0],x0.shape[1])
                
            ref_U = reference_U(x_dd, x_dv, x_dh)
            y_U = target_U(img, x0)
            N=15
            t = np.arange(-N,N+1)
            h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
            h_2d = h*np.transpose(h)
            im_pred_flat, p_u = coef_U(img, ref_U, h_2d)

    else:

        if dynamic == 0:
            
            diag_pred = model_p3.predict(ref_P3, batch_size = ref_P3.shape[0])
            
            if lossy:
                x_dd = (y_P3 - diag_pred).reshape(x0.shape[0],x0.shape[1])
            else:
                x_dd = (y_P3 - np.round(diag_pred)).reshape(x0.shape[0],x0.shape[1])
                
            if trans == '42_AG':
                ref_P1 = reference_P1_42_2D_AG(x0, x_dd)
                ref_P2 = reference_P2_42_2D_AG(x0, x1, x_dd)
                print('ref P1 P2 42_AG')
            else:
                ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
                
            y_P1, y_P2 = target_P1_P2(x1,x2)
            vertic_pred = model_p2.predict(ref_P2, batch_size = ref_P2.shape[0])
            horiz_pred = model_p1.predict(ref_P1, batch_size = ref_P1.shape[0])
            
            if lossy:
                
                x_dv = (y_P2 - vertic_pred).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - horiz_pred).reshape(x0.shape[0],x0.shape[1])
                
            else:
                
                x_dv = (y_P2 - np.round(vertic_pred)).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - np.round(horiz_pred)).reshape(x0.shape[0],x0.shape[1])
                
            ref_U = reference_U(x_dd, x_dv, x_dh)
            y_U = target_U(img, x0)
            im_pred_flat = model_u.predict(ref_U, batch_size = ref_U.shape[0])
            
        else:
            
            diag_pred, p3 = adaptive_predict(model_p3, ref_P3, y_P3, 0)
            
            if lossy:
                x_dd = (y_P3 - diag_pred).reshape(x0.shape[0],x0.shape[1])
            else:
                x_dd = (y_P3 - np.round(diag_pred)).reshape(x0.shape[0],x0.shape[1])
                
            if trans == '42_AG':
                ref_P1 = reference_P1_42_2D_AG(x0, x_dd)
                ref_P2 = reference_P2_42_2D_AG(x0, x1, x_dd)
                print('ref P1 P2 42_AG')
            else:
                ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
                
            y_P1, y_P2 = target_P1_P2(x1,x2)
            vertic_pred, p2 = adaptive_predict(model_p2, ref_P2, y_P2, 0)
            horiz_pred, p1 = adaptive_predict(model_p1, ref_P1, y_P1, 0)
            
            if lossy:
                
                x_dv = (y_P2 - vertic_pred).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - horiz_pred).reshape(x0.shape[0],x0.shape[1])
                
            else:
                
                x_dv = (y_P2 - np.round(vertic_pred)).reshape(x0.shape[0],x0.shape[1])
                x_dh = (y_P1 - np.round(horiz_pred)).reshape(x0.shape[0],x0.shape[1])
                
            ref_U = reference_U(x_dd, x_dv, x_dh)
            y_U = target_U(img, x0)
            im_pred_flat, p_u = adaptive_predict(model_u, ref_U, y_U, 0)

    decomp_img = np.zeros((img.shape[0],img.shape[1]))
    x_u = im_pred_flat.reshape(x0.shape[0],x0.shape[1])
    
    if lossy:
        x_smooth = x0 + x_u
    else:
        x_smooth = x0 + np.round(x_u)

        
    decomp_img[:img.shape[0]//2,:img.shape[1]//2] = x_smooth
    decomp_img[:img.shape[0]//2,img.shape[1]//2:] = x_dh
    decomp_img[img.shape[0]//2:,:img.shape[1]//2] = x_dv
    decomp_img[img.shape[0]//2:,img.shape[1]//2:] = x_dd
    
    if dynamic == 0:
        
        p = np.zeros((4))
        return decomp_img, p
        
    else:

        p = [p1, p2, p3, p_u]
        return decomp_img, p


def Backward(decomp_img, p, dynamic, lossy, trans, model_p1, model_p2, model_p3, model_u):
 
    num_pixels = decomp_img.shape[0]*decomp_img.shape[1]//4
    x_smooth = decomp_img[:decomp_img.shape[0]//2,:decomp_img.shape[1]//2]
    x_dh = decomp_img[:decomp_img.shape[0]//2,decomp_img.shape[1]//2:]
    x_dv = decomp_img[decomp_img.shape[0]//2:,:decomp_img.shape[1]//2]
    x_dd = decomp_img[decomp_img.shape[0]//2:,decomp_img.shape[1]//2:]
    ref_U = reference_U(x_dd, x_dv, x_dh)
    print(decomp_img.shape)

    #generate x0
    if model_p3 == 0:
        
        if dynamic == 0:
            im_pred_flat = coef_fixe_U(ref_U)
        else:
            im_pred_flat = np.expand_dims(np.sum(np.multiply(ref_U,p[3]),axis=1),axis=1)
            
    else:
        
        if dynamic == 0:
            im_pred_flat = model_u.predict(ref_U, batch_size = ref_U.shape[0])
        else:
            im_pred_flat, p_u = adaptive_predict(model_u, ref_U, 0, p[3])
        
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    if lossy:
        x0 = x_smooth - im_pred
    else:
        x0 = x_smooth - np.round(im_pred)
    if trans == '42_AG':
        ref_P1 = reference_P1_42_2D_AG(x0, x_dd)
        print('refP1 42_AG')
    else:
        ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
    
    #generate x1
    if model_p3 == 0:
        
        if dynamic == 0:
            im_pred_flat = coef_fixe_P_h_v(ref_P1)
        else:
            im_pred_flat = np.expand_dims(np.sum(np.multiply(ref_P1,p[0]),axis=1),axis=1)
            
    else:
        
        if dynamic == 0:
            im_pred_flat = model_p1.predict(ref_P1, batch_size = ref_P1.shape[0])
        else:
            im_pred_flat, p1 = adaptive_predict(model_p1, ref_P1, 0, p[0])

    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    if lossy:
        x1 = x_dh + im_pred
    else:
        x1 = x_dh + np.round(im_pred)

    #generate x2
    if trans == '42_AG':
        ref_P2 = reference_P2_42_2D_AG(x0, x1, x_dd)
        print('refP2 42_AG')
    if model_p3 == 0:
        
        if dynamic == 0:
            im_pred_flat = coef_fixe_P_h_v(ref_P2)
        else:
            im_pred_flat = np.expand_dims(np.sum(np.multiply(ref_P2,p[1]),axis=1),axis=1)
            
    else:
        
        if dynamic == 0:
            im_pred_flat = model_p2.predict(ref_P2, batch_size = ref_P2.shape[0])
        else:
            im_pred_flat, p2 = adaptive_predict(model_p2, ref_P2, 0, p = p[1])
            
            
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    
    if lossy:
        x2 = x_dv + im_pred
    else:
        x2 = x_dv + np.round(im_pred)

    
    #generate reference vector for P(HH)
    ref_P3 = Ref_P3(trans, x0, x1, x2)

    #generate x3
    if model_p3 == 0:
        
        if dynamic == 0:
            im_pred_flat = coef_fixe_P_d(ref_P3)
        else:
            im_pred_flat = np.expand_dims(np.sum(np.multiply(ref_P3,p[2]),axis=1),axis=1)
            
    else:
        
        if dynamic == 0:
            im_pred_flat = model_p3.predict(ref_P3, batch_size = ref_P3.shape[0])
        else:
            im_pred_flat, p3 = adaptive_predict(model_p3, ref_P3, 0, p = p[2])
            
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    
    if lossy:
        x3 = x_dd + im_pred
    else:
        x3 = x_dd + np.round(im_pred)
    
    #merge x0, x1, x2, and x3
    origin_img = np.zeros((decomp_img.shape[0],decomp_img.shape[1]))
    origin_img[::2,::2] = x0
    origin_img[::2,1::2] = x1
    origin_img[1::2,::2] = x2
    origin_img[1::2,1::2] = x3
    
    return origin_img

def Forward_par(img, lossy, model_X1, model_X2, model_X3, model_U):
    
    if lossy == False:
        img = np.round(img)
        
    x0 = img[::2,::2]
    x1 = img[::2,1::2]
    x2 = img[1::2,::2]
    x3 = img[1::2,1::2]
    num_pixels = x0.shape[0]*x0.shape[1]
    #print(img.shape)
    
    # generate reference vector for P(HH)
    ref_P3 = reference_P3_42_AG_par(x0, x1, x2)
    y_P3 = target_P3_par(x3)
    
    diag_pred = model_X3(ref_P3)
    diag_pred = diag_pred.numpy()
    
    if lossy:
        x_dd = (y_P3 - diag_pred).reshape((x0.shape[0],x0.shape[1]))
    else:
        x_dd = (y_P3 - np.round(diag_pred)).reshape((x0.shape[0],x0.shape[1]))
                
    ref_P1 = reference_P1_42_2D_AG_par(x0, x_dd)
    ref_P2 = reference_P2_42_2D_AG_par(x0, x1, x_dd)

    y_P1, y_P2 = target_P1_P2(x1,x2)

    vertic_pred = model_X2(ref_P2)
    horiz_pred = model_X1(ref_P1)
    vertic_pred = vertic_pred.numpy()
    horiz_pred = horiz_pred.numpy()
    
    if lossy:

        x_dv = (y_P2 - vertic_pred).reshape((x0.shape[0],x0.shape[1]))
        x_dh = (y_P1 - horiz_pred).reshape((x0.shape[0],x0.shape[1]))

    else:

        x_dv = (y_P2 - np.round(vertic_pred)).reshape((x0.shape[0],x0.shape[1]))
        x_dh = (y_P1 - np.round(horiz_pred)).reshape((x0.shape[0],x0.shape[1]))

    ref_U = reference_U_par(x_dd, x_dv, x_dh)
    y_U = target_U_par(img, x0)
    
    im_pred_flat = model_U(ref_U)
    im_pred_flat = im_pred_flat.numpy()
    decomp_img = np.zeros((img.shape[0],img.shape[1]))
    x_u = im_pred_flat.reshape((x0.shape[0],x0.shape[1]))
    
    if lossy:
        x_smooth = x0 + x_u
    else:
        x_smooth = x0 + np.round(x_u)
        
    decomp_img[:img.shape[0]//2,:img.shape[1]//2] = x_smooth[:,:]
    decomp_img[:img.shape[0]//2,img.shape[1]//2:] = x_dh[:,:]
    decomp_img[img.shape[0]//2:,:img.shape[1]//2] = x_dv[:,:]
    decomp_img[img.shape[0]//2:,img.shape[1]//2:] = x_dd[:,:]
    
    return decomp_img#, x_dh, x_dv, x_dd, x_smooth

def Backward_par(decomp_img, lossy, model_X1, model_X2, model_X3, model_U):
 
    num_pixels = decomp_img.shape[0]*decomp_img.shape[1]//4
    x_smooth = decomp_img[:decomp_img.shape[0]//2,:decomp_img.shape[1]//2]
    x_dh = decomp_img[:decomp_img.shape[0]//2,decomp_img.shape[1]//2:]
    x_dv = decomp_img[decomp_img.shape[0]//2:,:decomp_img.shape[1]//2]
    x_dd = decomp_img[decomp_img.shape[0]//2:,decomp_img.shape[1]//2:]
    ref_U = reference_U_par(x_dd, x_dv, x_dh)
    #print(decomp_img.shape)

    #generate x0
    im_pred_flat = model_U(ref_U)
    im_pred_flat = im_pred_flat.numpy()
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    if lossy:
        x0 = x_smooth - im_pred
    else:
        x0 = x_smooth - np.round(im_pred)
        
    ref_P1 = reference_P1_42_2D_AG_par(x0, x_dd)
    
    #generate x1
    im_pred_flat = model_X1(ref_P1)
    im_pred_flat = im_pred_flat.numpy()
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    if lossy:
        x1 = x_dh + im_pred
    else:
        x1 = x_dh + np.round(im_pred)

    #generate x2
    
    ref_P2 = reference_P2_42_2D_AG_par(x0, x1, x_dd)
    im_pred_flat = model_X2(ref_P2)
    im_pred_flat = im_pred_flat.numpy()
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    
    if lossy:
        x2 = x_dv + im_pred
    else:
        x2 = x_dv + np.round(im_pred)

    
    #generate reference vector for P(HH)
    ref_P3 = reference_P3_42_AG_par(x0, x1, x2)

    #generate x3
    im_pred_flat = model_X3(ref_P3)
    im_pred_flat = im_pred_flat.numpy()        
    im_pred = im_pred_flat.reshape(decomp_img.shape[0]//2,decomp_img.shape[1]//2)
    
    if lossy:
        x3 = x_dd + im_pred
    else:
        x3 = x_dd + np.round(im_pred)
    
    #merge x0, x1, x2, and x3
    origin_img = np.zeros((decomp_img.shape[0],decomp_img.shape[1]))
    origin_img[::2,::2] = x0
    origin_img[::2,1::2] = x1
    origin_img[1::2,::2] = x2
    origin_img[1::2,1::2] = x3
    
    return origin_img

def analysis(image, levels, dynamic, lossy, transform, *models):
    
    (H, W) = image.shape
    Id = np.zeros((H, W))
    app0 = image[:,:]
    m = 0
    p = []
    for level in range(levels):
        tmp = app0
        decomp_img, a = Forward(tmp, dynamic, lossy, transform, 
                                models[m+0], models[m+1], models[m+2], models[m+3])
        m = m+4
        p.append(a)
        app0 = decomp_img[:decomp_img.shape[0]//2,:decomp_img.shape[1]//2]
        Id[:H//(2**level),:W//(2**level)] = decomp_img[:,:]
    return Id, p


def synthesis(dec_img,
              levels,  
              p, 
              dynamic, 
              lossy,
              transform,
              *models):
    
    (H, W) = dec_img.shape
    origin_img = np.zeros((H, W))
    H = H//(2**levels)
    W = W//(2**levels)
    origin_img[:,:] = dec_img[:,:]
    m = 4*(levels-1)
    for level in range(levels):

        H, W = H*2, W*2
        Is = origin_img[:H,:W]
        
        origin_img[:H,:W] = Backward(Is, p[levels-level-1], dynamic, lossy, transform,
                                     models[m+0], models[m+1], models[m+2], models[m+3])
        m = m - 4
    return origin_img

def analysis_par(image, levels, lossy, *models):
    
    (H, W) = image.shape
    Id = np.zeros((H, W))
    app0 = image[:,:]
    m = 0
    for level in range(levels):
        tmp = app0
        decomp_img = Forward_par(tmp, lossy,
                                 models[m+0], models[m+1], models[m+2], models[m+3])
        m = m+4
        app0 = decomp_img[:decomp_img.shape[0]//2,:decomp_img.shape[1]//2]
        Id[:H//(2**level),:W//(2**level)] = decomp_img[:,:]
    return Id

def synthesis_par(dec_img,
              levels,
              lossy,
              *models):
    
    (H, W) = dec_img.shape
    origin_img = np.zeros((H, W))
    H = H//(2**levels)
    W = W//(2**levels)
    origin_img[:,:] = dec_img[:,:]
    m = 4*(levels-1)
    for level in range(levels):

        H, W = H*2, W*2
        Is = origin_img[:H,:W]
        
        origin_img[:H,:W] = Backward_par(Is, lossy,
                                         models[m+0], models[m+1], models[m+2], models[m+3])
        m = m - 4
    return origin_img

def compute_weights_par(orig_img, dec_img, levels, lossy, *models, br0, br1):
    
    W = np.zeros((levels,3))
    (M,N) = dec_img.shape
    wei_mat_moy = np.zeros((M,N))
    Is0 = np.zeros((M,N))
    Is1 = np.zeros((M,N))
    Is2 = np.zeros((M,N))
    Is3 = np.zeros((M,N))
    
    nbr_realisation = 1
    
    for t in range(nbr_realisation):
        
        wei_mat = np.zeros((M,N))
    
        Ms = M//(2**levels)
        Ns = N//(2**levels)
        br0 = br0[:Ms,:Ns]
        Is0[:,:] = dec_img[:,:]
        Is0[:Ms,:Ns] = dec_img[:Ms,:Ns]+br0
        Irec0 = synthesis_par(Is0, levels, lossy, *models)
        w_ap = skimage.metrics.mean_squared_error(orig_img,Irec0)*(4**levels)/(np.var(br0))
        wei_mat[:Ms,:Ns] = math.sqrt(w_ap)
    
        for i in range(levels):
            Is1[:,:] = dec_img[:,:] 
            Is1[:Ms,Ns:Ns*2] = dec_img[:Ms,Ns:Ns*2]+br1[:Ms,:Ns]
            Irec1 = synthesis_par(Is1, levels, lossy, *models)
            a1 = skimage.metrics.mean_squared_error(orig_img,Irec1)*(4**(levels-i))/(np.var(br1))
            W[levels-i-1,0] = a1
            wei_mat[:Ms,Ns:Ns*2] = math.sqrt(a1)
             
            
            Is2[:,:] = dec_img[:,:]
            Is2[Ms:Ms*2,:Ns] = dec_img[Ms:Ms*2,:Ns]+br1[:Ms,:Ns]
            Irec2 = synthesis_par(Is2, levels, lossy, *models)
            a2 = skimage.metrics.mean_squared_error(orig_img,Irec2)*(4**(levels-i))/(np.var(br1))
            W[levels-i-1,1] = a2
            wei_mat[Ms:Ms*2,:Ns] = math.sqrt(a2)
            

            Is3[:,:] = dec_img[:,:]
            Is3[Ms:Ms*2,Ns:Ns*2] = dec_img[Ms:Ms*2,Ns:Ns*2]+br1[:Ms,:Ns]
            Irec3 = synthesis_par(Is3, levels, lossy, *models)
            a3 = skimage.metrics.mean_squared_error(orig_img,Irec3)*(4**(levels-i))/(np.var(br1))
            W[levels-i-1,2] = a3
            wei_mat[Ms:Ms*2,Ns:Ns*2] = math.sqrt(a3)
                  
            Ms = 2*Ms
            Ns = 2*Ns
            
        wei_mat_moy = wei_mat_moy + wei_mat
        
    wei_mat_moy = wei_mat_moy / nbr_realisation
    
    dec_img_pond = np.multiply(dec_img, wei_mat_moy)
    
    return W, w_ap

def compute_weights(orig_img, dec_img, levels, p, dynamic, lossy, transform, *models):
    
    W = np.zeros((levels,3))
    (M,N) = dec_img.shape
    wei_mat_moy = np.zeros((M,N))
    Is0 = np.zeros((M,N))
    Is1 = np.zeros((M,N))
    Is2 = np.zeros((M,N))
    Is3 = np.zeros((M,N))
    
    nbr_realisation = 1
    
    for t in range(nbr_realisation):
        
        wei_mat = np.zeros((M,N))
    
        Ms = M//(2**levels)
        Ns = N//(2**levels)

        br0 =  np.random.normal(loc=0.0, scale=1.0, size=(Ms,Ns))
        
        Is0[:,:] = dec_img[:,:]
        Is0[:Ms,:Ns] = dec_img[:Ms,:Ns]+br0
        Irec0 = synthesis(Is0, levels, p, dynamic, lossy, transform, *models)
        w_ap = skimage.metrics.mean_squared_error(orig_img,Irec0)*(4**levels)/(np.var(br0))
        wei_mat[:Ms,:Ns] = math.sqrt(w_ap)
    
        for i in range(levels):
            
            br1 =  np.random.normal(loc=0.0, scale=1.0, size=(Ms,Ns))
            
            Is1[:,:] = dec_img[:,:] 
            Is1[:Ms,Ns:Ns*2] = dec_img[:Ms,Ns:Ns*2]+br1
            Irec1 = synthesis(Is1, levels, p, dynamic, lossy, transform, *models)
            a1 = skimage.metrics.mean_squared_error(orig_img,Irec1)*(4**(levels-i))/(np.var(br1))
            W[levels-i-1,0] = a1
            wei_mat[:Ms,Ns:Ns*2] = math.sqrt(a1)
             
            
            Is2[:,:] = dec_img[:,:]
            Is2[Ms:Ms*2,:Ns] = dec_img[Ms:Ms*2,:Ns]+br1
            Irec2 = synthesis(Is2, levels, p, dynamic, lossy, transform, *models)
            a2 = skimage.metrics.mean_squared_error(orig_img,Irec2)*(4**(levels-i))/(np.var(br1))
            W[levels-i-1,1] = a2
            wei_mat[Ms:Ms*2,:Ns] = math.sqrt(a2)
            

            Is3[:,:] = dec_img[:,:]
            Is3[Ms:Ms*2,Ns:Ns*2] = dec_img[Ms:Ms*2,Ns:Ns*2]+br1
            Irec3 = synthesis(Is3, levels, p, dynamic, lossy, transform, *models)
            a3 = skimage.metrics.mean_squared_error(orig_img,Irec3)*(4**(levels-i))/(np.var(br1))
            W[levels-i-1,2] = a3
            wei_mat[Ms:Ms*2,Ns:Ns*2] = math.sqrt(a3)
                  
            Ms = 2*Ms
            Ns = 2*Ns
            
        wei_mat_moy = wei_mat_moy + wei_mat
        
    wei_mat_moy = wei_mat_moy / nbr_realisation
    
    dec_img_pond = np.multiply(dec_img, wei_mat_moy)
    
    return W, w_ap

def weight_mat(image, W, w_ap, levels):
    (M,N) = image.shape
    wei_mat = np.zeros((M,N))
    Ms = M//(2**levels)
    Ns = N//(2**levels)
    wei_mat[:Ms,:Ns] = math.sqrt(w_ap)
    for i in range(levels):
        wei_mat[:Ms,Ns:Ns*2] = math.sqrt(W[levels-i-1,0])
        wei_mat[Ms:Ms*2,:Ns] = math.sqrt(W[levels-i-1,1])
        wei_mat[Ms:Ms*2,Ns:Ns*2] = math.sqrt(W[levels-i-1,2])
        Ms = 2*Ms
        Ns = 2*Ns
    return wei_mat


def compute_weights_1(orig_img, dec_img, levels, p, dynamic, lossy, transform, *models):
    
    W = np.zeros((levels,3))
    (M,N) = dec_img.shape
    Is0 = np.zeros((M,N))
    Is1 = np.zeros((M,N))
    Is2 = np.zeros((M,N))
    Is3 = np.zeros((M,N))
    L_ = 1
    epsilon = 0.02

    Ms = M//(2**levels)
    Ns = N//(2**levels)
    
    g_b = 0
    g = 1
    L = 1000
    m = 1
    Is0[:,:] = dec_img[:,:]
    while (L/L_ > 1 + epsilon) or (L_/L > 1 + epsilon) :
        print('t =', 1)
        n =  np.random.normal(loc=0.0, scale=1.0, size=(Ms,Ns))
        Is0[:Ms,:Ns] = dec_img[:Ms,:Ns] + (n / g)
        Irec0 = synthesis(Is0, levels, p, dynamic, lossy, transform, *models)
        #L = skimage.metrics.mean_squared_error(orig_img, Irec0)
        L = np.sum(np.square(orig_img - Irec0))
        print(L)
        if L/L_ > 1 + epsilon:
            g_b = g
            g = g * np.sqrt(L/L_)
        elif L_/L > 1 + epsilon:
            g = (g_b + g)/2
        print(g)
    w_ap = g
    for i in range(levels):
        g_b = 0
        g = 1
        L = 1000
        Is1[:,:] = dec_img[:,:] 
        while (L/L_ > 1 + epsilon) or (L_/L > 1 + epsilon) :
            print('t =', m+1)
            n1 =  np.random.normal(loc=0.0, scale = 1.0, size=(Ms,Ns))
            Is1[:Ms,Ns:Ns*2] = dec_img[:Ms,Ns:Ns*2] + (n1/g)
            Irec1 = synthesis(Is1, levels, p, dynamic, lossy, transform, *models)
            #L = skimage.metrics.mean_squared_error(orig_img,Irec1)
            L = np.sum(np.square(orig_img - Irec1))
            print(L)
            if L/L_ > 1 + epsilon:
                g_b = g
                g = g * np.sqrt(L/L_)
            elif L_/L > 1 + epsilon:
                g = (g_b + g)/2
            print(g)
        W[levels-i-1,0] = g
        
        g_b = 0
        g = 1
        L = 1000
        Is2[:,:] = dec_img[:,:]
        while (L/L_ > 1 + epsilon) or (L_/L > 1 + epsilon):
            print('t =', m+2)
            n2 =  np.random.normal(loc=0.0, scale=1.0, size=(Ms,Ns))
            Is2[Ms:Ms*2,:Ns] = dec_img[Ms:Ms*2,:Ns] + (n2/g)
            Irec2 = synthesis(Is2, levels, p, dynamic, lossy, transform, *models)
            #L = skimage.metrics.mean_squared_error(orig_img,Irec2)
            L = np.sum(np.square(orig_img - Irec2))
            print(L)
            if L/L_ > 1 + epsilon:
                g_b = g
                g = g * np.sqrt(L/L_)
            elif L_/L > 1 + epsilon:
                g = (g_b + g)/2
            print(g)
        W[levels-i-1,1] = g 
        
        g_b = 0
        g = 1
        L = 1000
        Is3[:,:] = dec_img[:,:]
        while (L/L_ > 1 + epsilon) or (L_/L > 1 + epsilon):
            print('t =', m+3)
            n3 =  np.random.normal(loc=0.0, scale=1.0, size = (Ms,Ns))
            Is3[Ms:Ms*2,Ns:Ns*2] = dec_img[Ms:Ms*2,Ns:Ns*2] + (n3/g)
            Irec3 = synthesis(Is3, levels, p, dynamic, lossy, transform, *models)
            #L = skimage.metrics.mean_squared_error(orig_img,Irec3)
            L = np.sum(np.square(orig_img - Irec3))
            print(L)
            if L/L_ > 1 + epsilon:
                g_b = g
                g = g * np.sqrt(L/L_)
            elif L_/L > 1 + epsilon:
                g = (g_b + g)/2
            print(g)
        W[levels-i-1,2] = g

        Ms = 2*Ms
        Ns = 2*Ns
        m = m+3
    
    return W, w_ap