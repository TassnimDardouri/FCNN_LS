import math
import numpy as np
import skimage.measure
import _pickle

def compute_mse_psnr(decomp_img_lossy, levels):
    
    mse_P1 = np.zeros((1,levels))
    mse_P2 = np.zeros((1,levels))
    mse_P3 = np.zeros((1,levels))
    H, W = decomp_img_lossy.shape
    
    for l in range(levels):

        approx_lossy = decomp_img_lossy[:H//2**(l+1),:W//2**(l+1)] 
        x_dh_lossy = decomp_img_lossy[:H//2**(l+1),W//2**(l+1):2*W//2**(l+1)] 
        x_dv_lossy = decomp_img_lossy[H//2**(l+1):2*H//2**(l+1),:W//2**(l+1)]
        x_dd_lossy = decomp_img_lossy[H//2**(l+1):2*H//2**(l+1),W//2**(l+1):2*W//2**(l+1)]

        mse_P3[:,l] = (x_dd_lossy**2).mean()
        mse_P2[:,l] = (x_dv_lossy**2).mean()
        mse_P1[:,l] = (x_dh_lossy**2).mean()

    im_mse = 0
    im_psnr = 0
    for j in range(levels):
        im_mse = im_mse + (0.25**(j+1)) * mse_P3[:,j] 
        im_mse = im_mse + (0.25**(j+1)) * mse_P2[:,j] 
        im_mse = im_mse + (0.25**(j+1)) * mse_P1[:,j]
    im_psnr = 10 * math.log10((255**2)/im_mse)
    
    return im_mse, im_psnr

def compute_entropy(decomp_img_lossless, levels):
    
    entropy_P1 = np.zeros((1,levels))
    entropy_P2 = np.zeros((1,levels))
    entropy_P3 = np.zeros((1,levels))
    entropy_U = np.zeros((1,1))
    H, W = decomp_img_lossless.shape
    
    for l in range(levels):
        
        approx_lossless = decomp_img_lossless[:H//2**(l+1),:W//2**(l+1)] 
        x_dh_lossless = decomp_img_lossless[:H//2**(l+1),W//2**(l+1):2*W//2**(l+1)] 
        x_dv_lossless = decomp_img_lossless[H//2**(l+1):2*H//2**(l+1),:W//2**(l+1)]
        x_dd_lossless = decomp_img_lossless[H//2**(l+1):2*H//2**(l+1),W//2**(l+1):2*W//2**(l+1)]
        
        entropy_P3[:,l] = skimage.measure.shannon_entropy(x_dd_lossless, base=2)
        entropy_P2[:,l] = skimage.measure.shannon_entropy(x_dv_lossless, base=2)
        entropy_P1[:,l] = skimage.measure.shannon_entropy(x_dh_lossless, base=2)
        
        if l==levels-1:
            entropy_U[:] = skimage.measure.shannon_entropy(approx_lossless, base=2)
    
    im_entropy = 0

    for j in range(levels):
        im_entropy = im_entropy + (0.25**(j+1)) * entropy_P3[:,j] 
        im_entropy = im_entropy + (0.25**(j+1)) * entropy_P2[:,j] 
        im_entropy = im_entropy + (0.25**(j+1)) * entropy_P1[:,j]
        if j == levels-1:
            im_entropy = im_entropy + (0.25**(j+1)) * entropy_U[:,:]
    return im_entropy

def global_metrics(num_im, levels, path_mse, path_entropy):
    
    with open(path_mse,'rb') as f:
        mse_U, mse_P1, mse_P2, mse_P3 = _pickle.load(f)
    with open(path_entropy,'rb') as f:
        entropy_U, entropy_P1, entropy_P2, entropy_P3 = _pickle.load(f)
    
    im_entropy = np.zeros((num_im,1))
    im_mse = np.zeros((num_im,1))
    im_psnr = np.zeros((num_im,1))
    for i in range(num_im):
        for j in range(levels):
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P3[i,j] 
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P2[i,j] 
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P1[i,j] 
            if j == levels-1:
                im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_U[i,j]

            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P3[i,j] 
            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P2[i,j] 
            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P1[i,j] 
        im_psnr[i,:] = 10 * math.log10((255**2)/im_mse[i,:])
    return im_mse, im_psnr, im_entropy

def global_metrics_no_approx12(num_im, levels, path_mse, path_entropy):
    
    with open(path_mse,'rb') as f:
        mse_U, mse_P1, mse_P2, mse_P3 = _pickle.load(f)
    with open(path_entropy,'rb') as f:
        entropy_U, entropy_P1, entropy_P2, entropy_P3 = _pickle.load(f)
    
    im_entropy = np.zeros((num_im,1))
    im_mse = np.zeros((num_im,1))
    im_psnr = np.zeros((num_im,1))
    for i in range(num_im):
        for j in range(levels):
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P3[i,j] 
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P2[i,j] 
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P1[i,j] 
            if j == levels-1 and levels == 3:
                im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_U[i,j]

            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P3[i,j] 
            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P2[i,j] 
            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P1[i,j] 
        im_psnr[i,:] = 10 * math.log10((255**2)/im_mse[i,:])
    return im_mse, im_psnr, im_entropy

def global_metrics_single_lev(num_im, levels, path_mse, path_entropy):
    
    with open(path_mse,'rb') as f:
        mse_U, mse_P1, mse_P2, mse_P3 = _pickle.load(f)
    with open(path_entropy,'rb') as f:
        entropy_U, entropy_P1, entropy_P2, entropy_P3 = _pickle.load(f)
    
    im_entropy = np.zeros((num_im,1))
    im_mse = np.zeros((num_im,1))
    im_psnr = np.zeros((num_im,1))
    for i in range(num_im):
        for j in range(levels-1, levels):
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P3[i,j] 
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P2[i,j] 
            im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_P1[i,j] 
            if j == levels-1 and levels == 3:
                im_entropy[i,:] = im_entropy[i,:] + (0.25**(j+1)) * entropy_U[i,j]

            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P3[i,j] 
            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P2[i,j] 
            im_mse[i,:] = im_mse[i,:] + (0.25**(j+1)) * mse_P1[i,j] 
        im_psnr[i,:] = 10 * math.log10((255**2)/im_mse[i,:])
    return im_mse, im_psnr, im_entropy