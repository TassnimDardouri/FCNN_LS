from tensorflow.keras.models import load_model
import sys
import os
import glob
from skimage import io 
from multiprocessing import Pool
import numpy as np
import _pickle
import time

from utils.data_utils import subband
from utils.helpers import get_args
from utils.references import (Ref_P3, target_P3, 
                              Ref_P1_P2, target_P1_P2, 
                              reference_U, target_U, 
                              reference_P1_42_2D_AG, reference_P2_42_2D_AG)

from utils.model_utils import adaptive_predict, custom_loss, L_3_4, L_beta, L_06, L_const_beta, log_L_const_beta, round_mse
from utils.train_parser import Options

args  = get_args()

loss_dict = {L_06: 'L_06',
             L_3_4: 'L_3_4', 
             L_beta: 'L_beta',
             custom_loss:'custom_loss',
             log_L_const_beta:' log_L_const_beta',
             round_mse: 'round_mse'}

def create_directory(path):
    print(path)
    try:
        if os.path.isdir(path):
            print ("Directory exists")
            pass
        else:
            os.makedirs(path)
            print ("Successfully created the directory")
    except OSError:
        print ("Creation of the directory failed")
        return
        
        
def process_P3(args_list, levels = 3):
    
    im_path = args_list[0]
    ref_P3_path = args_list[1]
    trans = args_list[2]
    
    image = np.load(im_path)
    H, W = image.shape
    image = image[:(H//(2**levels))*(2**levels),:(W//(2**levels))*(2**levels)]
    H, W = image.shape

    x0, x1, x2, x3 = subband(image)
    ref_P3 = Ref_P3(trans, x0, x1, x2)
    y_P3 = target_P3(x3)
    
    ref_P3 = ref_P3.astype(np.float16)
    y_P3 = y_P3.astype(np.float16)
    
    image_name = os.path.basename(im_path)
    image_name = image_name.replace('.npy', '.pickle')

    save_file_path = os.path.join(ref_P3_path, image_name)
    with open(save_file_path, 'wb') as f:
        _pickle.dump([ref_P3, y_P3], f)

    sys.stdout.write('\r'+save_file_path)
    sys.stdout.flush()
        
def X3_input(opt):

    create_directory(opt.P3_train_path)
    create_directory(opt.P3_test_path)
    
    train_paths = [[os.path.join(opt.im_path_train, file), opt.P3_train_path, opt.trans] \
                   for file in os.listdir(opt.im_path_train) if '.npy' in file]

    test_paths = [[os.path.join(opt.im_path_test, file), opt.P3_test_path, opt.trans] \
                       for file in os.listdir(opt.im_path_test) if '.npy' in file]
    
    pool = Pool(opt.num_workers)
    pool.map(process_P3, train_paths)
    pool.map(process_P3, test_paths)
    pool.close()
    
def process_P1_P2(args_list, levels = 3):
    
    y_P3 = args_list['y_P3']
    im_path = args_list['im_path']
    P2_path = args_list['P2_path']
    P1_path = args_list['P1_path']
    pred = args_list['pred']
    trans = args_list['trans']
    
    image = np.load(im_path)
    H, W = image.shape
    image = image[:(H//(2**levels))*(2**levels),:(W//(2**levels))*(2**levels)]
    H, W = image.shape
    x0, x1, x2, x3 = subband(image)
    x_dd = (y_P3 - pred).reshape(x0.shape[0],x0.shape[1])
    
    if trans == '42_AG':
        ref_P1 = reference_P1_42_2D_AG(x0, x_dd)
        ref_P2 = reference_P2_42_2D_AG(x0, x1, x_dd)
    else:
        ref_P1, ref_P2 = Ref_P1_P2(trans, x0, x_dd)
        
    y_P1, y_P2 = target_P1_P2(x1, x2)
    
    ref_P1 = ref_P1.astype(np.float16)
    y_P1 = y_P1.astype(np.float16)
    ref_P2 = ref_P2.astype(np.float16)
    y_P2 = y_P2.astype(np.float16)
    
    image_name = os.path.basename(im_path)
    image_name = image_name.replace('.npy', '.pickle')

    save_file_path_1 = os.path.join(P1_path, image_name)
    save_file_path_2 = os.path.join(P2_path, image_name)
    
    with open(save_file_path_1, 'wb') as f:
        _pickle.dump([ref_P1, y_P1], f)
            
    with open(save_file_path_2, 'wb') as f:
        _pickle.dump([ref_P2, y_P2], f)
    
    sys.stdout.write('\r'+ save_file_path_1+ '\n' + save_file_path_2)
    sys.stdout.flush()
    
def load_input_data(path):
    with open(path, 'rb') as f:
        ref, y = _pickle.load(f)
    return [ref, y, path]

def X1_X2_input(args, opt, Loss = None, train = True, test = True):
    if train:
        create_directory(opt.P1_train_path)
        create_directory(opt.P2_train_path)
    if test:
        create_directory(opt.P1_test_path)
        create_directory(opt.P2_test_path)
    
    model_p3 = load_model(opt.P3_model_path, compile = False)
    """
    if args.Loss == 'L_const_beta':
        Loss = L_const_beta(float(args.beta_p3))
        model_p3 = load_model(opt.P3_model_path, custom_objects = {'Loss': Loss})
    elif Loss is None:
        model_p3 = load_model(opt.P3_model_path)
    else:
        model_p3 = load_model(opt.P3_model_path, custom_objects = {loss_dict[Loss]: Loss})
    """   
    if train:
        P3_train_paths = [os.path.join(opt.P3_train_path, file) for file \
                          in os.listdir(opt.P3_train_path) if '.pickle' in file]
    if test:
        P3_test_paths = [os.path.join(opt.P3_test_path, file) for file \
                          in os.listdir(opt.P3_test_path) if '.pickle' in file]

    #pool = Pool(opt.num_workers)
    #if train:
        #train_ref_P3_y_P3_path = pool.map(load_input_data, P3_train_paths)
    #if test:
        #test_ref_P3_y_P3_path = pool.map(load_input_data, P3_test_paths)
    #pool.close()
    if train:
        train_args_list = [] 
        #for ref_P3, y_P3, P3_path in train_ref_P3_y_P3_path:
        for P3_path in P3_train_paths:
            ref_P3, y_P3, P3_path = load_input_data(P3_path)
            print(P3_path)
            im_pred_flat = Predict(ref_P3, y_P3, model_p3, opt.dynamic)
            im_path = get_matching_im_path(opt, P3_path, 'train')

            train_args_list.append({'y_P3' :y_P3, 
                                  'pred':im_pred_flat, 
                                  'P2_path':opt.P2_train_path, 
                                  'P1_path':opt.P1_train_path, 
                                  'im_path':im_path,
                                  'trans':opt.trans})
    if test:
        test_args_list = []
        #for ref_P3, y_P3, P3_path in test_ref_P3_y_P3_path:
        for P3_path in P3_test_paths:
            ref_P3, y_P3, P3_path = load_input_data(P3_path)
            im_pred_flat = Predict(ref_P3, y_P3, model_p3, opt.dynamic)
            im_path = get_matching_im_path(opt, P3_path, 'test')

            test_args_list.append({'y_P3' :y_P3, 
                                  'pred':im_pred_flat, 
                                  'P2_path':opt.P2_test_path, 
                                  'P1_path':opt.P1_test_path, 
                                  'im_path':im_path,
                                  'trans':opt.trans})
        
    pool = Pool(opt.num_workers)
    if train:
        pool.map(process_P1_P2, train_args_list)
    if test:
        pool.map(process_P1_P2, test_args_list)
    pool.close()
    
def process_U(args_list, levels = 3):
    y_P3 = args_list['y_P3']
    diag_pred = args_list['P3_pred']
    y_P2 = args_list['y_P2'] 
    vertic_pred = args_list['P2_pred']
    y_P1 = args_list['y_P1']
    horiz_pred = args_list['P1_pred']
    U_path = args_list['U_path']
    im_path = args_list['im_path']
    
    image = np.load(im_path)
    H, W = image.shape
    image = image[:(H//(2**levels))*(2**levels),:(W//(2**levels))*(2**levels)]
    H, W = image.shape
    x0, x1, x2, x3 = subband(image)
    x_dd = (y_P3 - diag_pred).reshape(H//2,W//2)
    x_dv = (y_P2 - vertic_pred).reshape(H//2,W//2)
    x_dh = (y_P1 - horiz_pred).reshape(H//2,W//2)

    ref_U = reference_U(x_dd, x_dv, x_dh)
    y_U = target_U(image, x0)
    
    ref_U = ref_U.astype(np.float16)
    y_U = y_U.astype(np.float16)
    
    image_name = os.path.basename(im_path)
    image_name = image_name.replace('.npy', '.pickle')

    save_file_path = os.path.join(U_path, image_name)
    with open(save_file_path, 'wb') as f:
        _pickle.dump([ref_U, y_U], f)

    sys.stdout.write('\r'+save_file_path)
    sys.stdout.flush()

def U_input(args, opt, Loss = None, train = True, test = True):
    model_p1 = load_model(opt.P1_model_path, compile = False)
    model_p2 = load_model(opt.P2_model_path, compile = False)
    model_p3 = load_model(opt.P3_model_path, compile = False)
    """
    if args.Loss == 'L_const_beta':
        
        Loss_p1 = L_const_beta(float(args.beta_p1))
        Loss_p2 = L_const_beta(float(args.beta_p2))
        Loss_p3 = L_const_beta(float(args.beta_p3))
        
        model_p1 = load_model(opt.P1_model_path, custom_objects = {'Loss_p1':Loss_p1})
        model_p2 = load_model(opt.P2_model_path, custom_objects = {'Loss_p2':Loss_p2})
        model_p3 = load_model(opt.P3_model_path, custom_objects = {'Loss_p3':Loss_p3})
        
    elif Loss is None:
        
        model_p1 = load_model(opt.P1_model_path)
        model_p2 = load_model(opt.P2_model_path)
        model_p3 = load_model(opt.P3_model_path)
        
    else:
        
        model_p1 = load_model(opt.P1_model_path, custom_objects = {loss_dict[Loss]:Loss})
        model_p2 = load_model(opt.P2_model_path, custom_objects = {loss_dict[Loss]:Loss})
        model_p3 = load_model(opt.P3_model_path, custom_objects = {loss_dict[Loss]:Loss})
    """
    model_p1.summary()
    model_p2.summary()
    model_p3.summary()
    if train:
        create_directory(opt.U_train_path)
    if test:
        create_directory(opt.U_test_path)
        
    if train:
        P3_train_paths = [os.path.join(opt.P3_train_path, file) for file \
                          in os.listdir(opt.P3_train_path) if '.pickle' in file]
        P2_train_paths = [os.path.join(opt.P2_train_path, file) for file \
                          in os.listdir(opt.P2_train_path) if '.pickle' in file]
        P1_train_paths = [os.path.join(opt.P1_train_path, file) for file \
                          in os.listdir(opt.P1_train_path) if '.pickle' in file]
        
    if test:
        P3_test_paths = [os.path.join(opt.P3_test_path, file) for file \
                          in os.listdir(opt.P3_test_path) if '.pickle' in file]
        P2_test_paths = [os.path.join(opt.P2_test_path, file) for file \
                          in os.listdir(opt.P2_test_path) if '.pickle' in file]
        P1_test_paths = [os.path.join(opt.P1_test_path, file) for file \
                          in os.listdir(opt.P1_test_path) if '.pickle' in file]
        
    if train:
        P3_train_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        P2_train_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        P1_train_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    
    if test:
        P3_test_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        P2_test_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        P1_test_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    if train:
        train_args_list = []
        for P3_path, P2_path, P1_path in zip(P3_train_paths, P2_train_paths, P1_train_paths):
            ref_P3, y_P3, P3_path = load_input_data(P3_path)
            ref_P2, y_P2, P2_path = load_input_data(P2_path)
            ref_P1, y_P1, P1_path = load_input_data(P1_path)
            print(P3_path, P2_path, P1_path)

            diag_pred = Predict(ref_P3, y_P3, model_p3, opt.dynamic)
            vertic_pred = Predict(ref_P2, y_P2, model_p2, opt.dynamic)
            horiz_pred = Predict(ref_P1, y_P1, model_p1, opt.dynamic)

            im_path = get_matching_im_path(opt, P1_path, 'train')

            train_args_list.append({'y_P3' :y_P3, 'P3_pred':diag_pred,
                                       'y_P2' :y_P2, 'P2_pred':vertic_pred,
                                       'y_P1' :y_P1, 'P1_pred':horiz_pred,
                                       'U_path':opt.U_train_path, 'im_path':im_path})
    if test:
        test_args_list = [] 

        for P3_path, P2_path, P1_path in zip(P3_test_paths, P2_test_paths, P1_test_paths):
            ref_P3, y_P3, P3_path = load_input_data(P3_path)
            ref_P2, y_P2, P2_path = load_input_data(P2_path)
            ref_P1, y_P1, P1_path = load_input_data(P1_path)
            print(P3_path, P2_path, P1_path)

            diag_pred = Predict(ref_P3, y_P3, model_p3, opt.dynamic)
            vertic_pred = Predict(ref_P2, y_P2, model_p2, opt.dynamic)
            horiz_pred = Predict(ref_P1, y_P1, model_p1, opt.dynamic)


            im_path = get_matching_im_path(opt, P1_path, 'test')

            test_args_list.append({'y_P3' :y_P3, 'P3_pred':diag_pred,
                                   'y_P2' :y_P2, 'P2_pred':vertic_pred,
                                   'y_P1' :y_P1, 'P1_pred':horiz_pred,
                                   'U_path':opt.U_test_path, 'im_path':im_path})

    pool = Pool(opt.num_workers)
    if train:
        pool.map(process_U, train_args_list)
    if test:
        pool.map(process_U, test_args_list)
    pool.close()
    
def approx(opt, Loss = None):
    
    create_directory(opt.approx_path_train)
    create_directory(opt.approx_path_test)
    
    model_u = load_model(opt.U_model_path, compile = False)
    """
    if Loss is None:
        model_u = load_model(opt.U_model_path)
    else:
        model_u = load_model(opt.U_model_path, custom_objects = {loss_dict[Loss]:Loss})
    """
    U_train_paths = [os.path.join(opt.U_train_path, file) for file \
                      in os.listdir(opt.U_train_path) if '.pickle' in file]

    U_test_paths = [os.path.join(opt.U_test_path, file) for file \
                      in os.listdir(opt.U_test_path) if '.pickle' in file]
    
    U_train_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    U_test_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    
    pool = Pool(opt.num_workers)
    train_ref_U_y_U_path = pool.map(load_input_data, U_train_paths)
    test_ref_U_y_U_path = pool.map(load_input_data, U_test_paths)
    pool.close()

    train_args_list = [] 
    for ref_U, y_U, U_path in train_ref_U_y_U_path:
        U_pred = Predict(ref_U, y_U, model_u, opt.dynamic)
        im_path = get_matching_im_path(opt, U_path, 'train')
        train_args_list.append({'U_pred':U_pred, 
                                'approx_path':opt.approx_path_train, 
                                'im_path':im_path})
   
    test_args_list = [] 
    for ref_U, y_U, U_path in test_ref_U_y_U_path:
        U_pred = Predict(ref_U, y_U, model_u, opt.dynamic)
        im_path = get_matching_im_path(opt, U_path, 'test')
        test_args_list.append({'U_pred':U_pred, 
                               'approx_path':opt.approx_path_test, 
                               'im_path':im_path})
        
    pool = Pool(opt.num_workers)
    pool.map(process_approx, test_args_list)
    pool.map(process_approx, train_args_list)
    pool.close()

def process_approx(args_list, levels = 3):
    U_pred = args_list['U_pred']
    approx_path = args_list['approx_path']
    im_path = args_list['im_path']
    
    image = np.load(im_path)
    H, W = image.shape
    image = image[:(H//(2**levels))*(2**levels),:(W//(2**levels))*(2**levels)]
    H, W = image.shape
    x0, x1, x2, x3 = subband(image)
    x_u = U_pred.reshape(x0.shape[0],x0.shape[1])
    approx = x0 + x_u
    
    image_name = os.path.basename(im_path)
    save_file_path = os.path.join(approx_path, image_name)
    np.save(save_file_path, approx)

    sys.stdout.write('\r'+save_file_path)
    sys.stdout.flush()
    
def Predict(x, y, model, dynamic = 0):
    if dynamic == 0:
        pred = model.predict(x, batch_size = x.shape[0])
    else:
        pred, p = adaptive_predict(model, x, y, 0)
    return pred

def get_matching_im_path(opt, pickle_path, data_folder):
    
    if data_folder == 'train':
        imdir_path = opt.im_path_train
        
    elif data_folder == 'test':
        imdir_path = opt.im_path_test
    else:
        raise Exception('please specify data folder')
        
    im_name = os.path.basename(pickle_path).replace('.pickle', '.npy')    
    im_path = os.path.join(imdir_path, im_name)
    return im_path
'''
if __name__ == '__main__':
    opt = Options(trans = '22', method = 'process_onepass_22', level = 1)
    p3_model_path = '/data/tdardour_data/image_comp/Fwd_ops/clic_dataset/onepass1_22/level1_X3_onepass_300.h5'
    p2_model_path = '/data/tdardour_data/image_comp/Fwd_ops/clic_dataset/onepass1_22/level1_X2_onepass_300.h5'
    p1_model_path = '/data/tdardour_data/image_comp/Fwd_ops/clic_dataset/onepass1_22/level1_X1_onepass_300.h5'
    u_model_path = '/data/tdardour_data/image_comp/Fwd_ops/clic_dataset/onepass1_22/level1_U_onepass_300.h5'
    opt.configure({'P3_final_model_path': p3_model_path,
                  'P2_final_model_path': p2_model_path,
                  'P1_final_model_path': p1_model_path,
                  'U_final_model_path': u_model_path,
                  'dynamic': 1})
    start = time.time()
    X3_input(opt)
    X1_X2_input(opt, Loss=custom_loss)
    U_input(opt, Loss=custom_loss)
    approx(opt, Loss=custom_loss)
    end = time.time()
    print(end-start)
'''