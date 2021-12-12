from keras import backend as k
from tensorflow.keras.models import load_model
import sys
import os
import glob
from skimage import io 

from utils.data_utils import load_pickle, dump_pickle, subband, get_num_im, rgb2gray
from utils.references import Ref_P3, target_P3, Ref_P1_P2, target_P1_P2, reference_U, target_U
from utils.model_utils import adaptive_predict, custom_loss, L_3_4, L_beta

loss_dict = {L_3_4: 'L_3_4', 
             L_beta: 'L_beta',
             custom_loss:'custom_loss'}

def create_directory(path):
    try:
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)
    except OSError:
        print ("Creation of the directory failed")
        return
    else:
        print ("Successfully created the directory")
        
def prev_X3_input(ref_P3_path, im_path, level_input_path, trans):
    if '/*.png' in im_path:
        num_im = get_num_im(im_path)
    else: 
        num_im = get_num_im(im_path+'/*.png')
    addrs = glob.glob(im_path)
    try:
        if os.path.isdir(ref_P3_path):
            pass
        else:
            os.mkdir(ref_P3_path)
    except OSError:
        print ("Creation of the directory failed")
        return
    else:
        print ("Successfully created the directory")

    for i in range(num_im):

        if level_input_path is None:
            if 'ordered' in im_path:
                path = im_path+ '/image'+str(i+1)+'.png'
                sys.stdout.write('\r'+path)
                sys.stdout.flush()
                image = io.imread(path)
                
            else:
                image = io.imread(addrs[i])
            if image is not None:
                image = rgb2gray(image)
                image = image[:(image.shape[0]//8)*8,:(image.shape[1]//8)*8]
        else:
            image = load_pickle(level_input_path, i)

        x0, x1, x2, x3 = subband(image)

        ref_P3 = Ref_P3(trans, x0, x1, x2)
        y_P3 = target_P3(x3)
        
        dump_pickle(ref_P3_path, i, (ref_P3, y_P3))

        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()
        
def prev_X1_X2_input(model_p3_path, im_path, level_input_path, ref_P1_path, ref_P2_path, trans, Loss, dynamic):
    try:
        if os.path.isdir(ref_P1_path):
            pass
        else:
            os.mkdir(ref_P1_path)
            
        if os.path.isdir(ref_P2_path):
            pass
        else:
            os.mkdir(ref_P2_path)
    except OSError:
        print ("Creation of the directory failed")
        return
    else:
        print ("Successfully created the directory")
    
    if Loss is None:
        model_p3 = load_model(model_p3_path)
    else:
        model_p3 = load_model(model_p3_path, custom_objects = {loss_dict[Loss]: Loss})
    
    if '/*.png' in im_path:
        num_im = get_num_im(im_path)
    else: 
        num_im = get_num_im(im_path+'/*.png')
    addrs = glob.glob(im_path)
    for i in range(num_im):
        if level_input_path is None:
            if 'ordered' in im_path:
                path = im_path+ '/image'+str(i+1)+'.png'
                sys.stdout.write('\r'+path)
                sys.stdout.flush()
                image = io.imread(path)
                
            else:
                image = io.imread(addrs[i])
            if image is not None:
                image = rgb2gray(image)
                image = image[:(image.shape[0]//8)*8,:(image.shape[1]//8)*8]
        else:
            image = load_pickle(level_input_path, i)

        x0, x1, x2, x3 = subband(image)
        ref_P3 = Ref_P3(trans, x0, x1, x2)
        y_P3 = target_P3(x3)

        if dynamic == 0:
            im_pred_flat = model_p3.predict(ref_P3, batch_size = ref_P3.shape[0])
        else:
            im_pred_flat, p3 = adaptive_predict(model_p3, ref_P3, y_P3, 0)

        x_dd = (y_P3 - im_pred_flat).reshape(x0.shape[0],x0.shape[1])

        ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
        y_P1, y_P2 = target_P1_P2(x1, x2)

        dump_pickle(ref_P1_path, i, (ref_P1,y_P1))
        dump_pickle(ref_P2_path, i, (ref_P2,y_P2))

        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()
        
def prev_U_input(model_p1_path, model_p2_path, model_p3_path, im_path, level_input_path, ref_u_path, trans, Loss, dynamic):
    
    if Loss is None:
        
        model_p1 = load_model(model_p1_path)
        model_p2 = load_model(model_p2_path)
        model_p3 = load_model(model_p3_path)
        
    else:
        
        model_p1 = load_model(model_p1_path, custom_objects = {loss_dict[Loss]:Loss})
        model_p2 = load_model(model_p2_path, custom_objects = {loss_dict[Loss]:Loss})
        model_p3 = load_model(model_p3_path, custom_objects = {loss_dict[Loss]:Loss})
    
    try:
        if os.path.isdir(ref_u_path):
            pass
        else:
            os.mkdir(ref_u_path)
    except OSError:
        print ("Creation of the directory failed")
        return
    else:
        print ("Successfully created the directory")
        
    if '/*.png' in im_path:
        num_im = get_num_im(im_path)
    else: 
        num_im = get_num_im(im_path+'/*.png')
    addrs = glob.glob(im_path)
    for i in range(num_im):
        if level_input_path is None:
            if 'ordered' in im_path:
                path = im_path+ '/image'+str(i+1)+'.png'
                sys.stdout.write('\r'+path)
                sys.stdout.flush()
                image = io.imread(path)
                
            else:
                image = io.imread(addrs[i])
            if image is not None:
                image = rgb2gray(image)
                image = image[:(image.shape[0]//8)*8,:(image.shape[1]//8)*8]
        else:
            image = load_pickle(level_input_path, i)

        x0, x1, x2, x3 = subband(image)

        ref_P3 = Ref_P3(trans, x0, x1, x2)
        y_P3 = target_P3(x3)

        if dynamic == 0:
            diag_pred = model_p3.predict(ref_P3, batch_size = ref_P3.shape[0])
        else:
            diag_pred, p3 = adaptive_predict(model_p3, ref_P3, y_P3, 0)

        x_dd = (y_P3 - diag_pred).reshape(image.shape[0]//2,image.shape[1]//2)

        ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
        y_P1, y_P2 = target_P1_P2(x1, x2)

        if dynamic == 0:

            vertic_pred = model_p2.predict(ref_P2, batch_size = ref_P2.shape[0])
            horiz_pred = model_p1.predict(ref_P1, batch_size = ref_P1.shape[0])

        else:

            vertic_pred, p2 = adaptive_predict(model_p2, ref_P2, y_P2, 0)
            horiz_pred, p1 = adaptive_predict(model_p1, ref_P1, y_P1, 0)

        x_dv = (y_P2 - vertic_pred).reshape(image.shape[0]//2,image.shape[1]//2)
        x_dh = (y_P1 - horiz_pred).reshape(image.shape[0]//2,image.shape[1]//2)

        ref_U = reference_U(x_dd, x_dv, x_dh)
        y_U = target_U(image, x0)

        dump_pickle(ref_u_path, i,(ref_U,y_U))
        
        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()
        
def prev_approx(model_u_path, ref_u_path, im_path, level_input_path, approx_path, trans, Loss, dynamic):
    
    try:
        if os.path.isdir(approx_path):
            pass
        else:
            os.mkdir(approx_path)
    except OSError:
        print ("Creation of the directory failed")
        return
    else:
        print ("Successfully created the directory")

    if Loss is None:
        model_u = load_model(model_u_path)
    else:
        model_u = load_model(model_u_path, custom_objects = {loss_dict[Loss]:Loss})
        
    if '/*.png' in im_path:
        num_im = get_num_im(im_path)
    else: 
        num_im = get_num_im(im_path+'/*.png')
        
    addrs = glob.glob(im_path)
    
    for i in range(num_im):
        
        if level_input_path is None:
            
            if 'ordered' in im_path:
                path = im_path+ '/image'+str(i+1)+'.png'
                sys.stdout.write('\r'+path)
                sys.stdout.flush()
                image = io.imread(path)
                
            else:
                image = io.imread(addrs[i])
            if image is not None:
                image = rgb2gray(image)
                image = image[:(image.shape[0]//8)*8,:(image.shape[1]//8)*8]
        else:
            image = load_pickle(level_input_path, i)

        x0 = image[::2,::2]
        (ref_U, y_U) = load_pickle(ref_u_path, i)

        if dynamic == 0:
            im_pred_flat = model_u.predict(ref_U, batch_size = ref_U.shape[0])
        else:
            im_pred_flat, p_u = adaptive_predict(model_u, ref_U, y_U, 0)

        x_u = im_pred_flat.reshape(x0.shape[0],x0.shape[1])
        approx = x0 + x_u
        dump_pickle(approx_path, i, approx)
        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()