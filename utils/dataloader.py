import numpy as np
import _pickle
from multiprocessing import Pool
import os
from tensorflow.keras.utils import Sequence
import math
from utils.data_utils import load_pickle, get_num_im

def get_data_length(dir_path, batch_size = None):
    if batch_size == None:
        data_length = len([file for file in os.listdir(dir_path) if 'pickle' in file])
        return data_length
    else:
        data_length = 0
        for file_path in [os.path.join(dir_path, file) \
                          for file in os.listdir(dir_path) if 'pickle' in file]:
            with open(file_path, 'rb') as f:
                ref, y = _pickle.load(f)
            if ref.shape[0] % batch_size == 0:
                data_length += ref.shape[0] // batch_size
            else:
                data_length += ref.shape[0] // batch_size + 1
        return data_length

def get_input_target(file_path):
    with open(file_path, 'rb') as f:
        batch_x, batch_y = _pickle.load(f)
    return [batch_x, batch_y]

def full_loader(opt, subband):
    
    if subband == 'P3':
        train_path = opt.P3_train_path
        test_path = opt.P3_test_path
        
    elif subband == 'P2':
        train_path = opt.P2_train_path
        test_path = opt.P2_test_path
        
    elif subband == 'P1':
        train_path = opt.P1_train_path
        test_path = opt.P1_test_path
        
    elif subband == 'U':
        train_path = opt.U_train_path
        test_path = opt.U_test_path
        
    else:
        raise ValueError('please specify subband')
        
    train_file_paths = [os.path.join(train_path, str(file_num)+'.pickle') for \
                        file_num in range(1, len(os.listdir(train_path))+1)]
    
    test_file_paths = [os.path.join(test_path, str(file_num)+'.pickle') for \
                       file_num in range(1, len(os.listdir(test_path))+1)]
    
    pool = Pool(opt.num_workers)
    full_test_x_y = pool.map(get_input_target, test_file_paths)
    full_train_x_y = pool.map(get_input_target, train_file_paths)
    pool.close()
    
    x_train = [full_train_x_y[i][0] for i in range(len(full_train_x_y))]
    x_train = np.concatenate(x_train, axis = 0)
    y_train = [full_train_x_y[i][1] for i in range(len(full_train_x_y))]
    y_train = np.concatenate(y_train, axis = 0)
    
    x_test = [full_test_x_y[i][0] for i in range(len(full_test_x_y))]
    x_test = np.concatenate(x_test, axis = 0)
    y_test = [full_test_x_y[i][1] for i in range(len(full_test_x_y))]
    y_test = np.concatenate(y_test, axis = 0)
    
    return x_train, y_train, x_test, y_test

def input_generator(n, im_path, batch_size = None):
    number = 1
    count = 0
    len_data = get_num_im(im_path+'/*.pickle')
    while 1:
        full_im_ref, full_im_y = load_pickle(im_path, number)
            
        count+=1
        number+=1
        if number == len_data:
            number = 1
        if count % n == 0:
            if batch_size == None:
                    yield (full_im_ref, full_im_y)
            else:
                for i in range(0, full_im_ref.shape[0], batch_size):
                    if i <= full_im_ref.shape[0]:
                        if i+batch_size <= full_im_ref.shape[0]:
                            batch_ref = full_im_ref[i:i+batch_size,:]
                            batch_y = full_im_y[i:i+batch_size,:]
                            yield (batch_ref, batch_y)
                        else:
                            batch_ref = full_im_ref[i:,:] 
                            batch_y = full_im_y[i:,:]
                            yield (batch_ref, batch_y)
            
            
def input_generator_beta(n, im_path, beta):
    beta = np.expand_dims(beta, axis =1)
    number = 0
    count = 0
    len_data = get_num_im(im_path+'/*.pickle')
    
    while 1:
        print(number)
        batch_ref, batch_y = load_pickle(im_path, number)
        H,W = batch_y.shape
        beta_i = np.zeros((H,W))
        if beta[number] > 0.4:
            beta_i[:,:] = beta[number]
        else:
            beta_i[:,:] = 0.4
        count+=1
        number+=1
        if number == len_data:
            number = 0
        if count % n == 0:
            yield (batch_ref, np.concatenate([batch_y, beta_i], axis = 1))
            
class generator_beta(Sequence):

    def __init__(self, x_set, beta, batch_size=1):
        self.x = x_set
        self.batch_size = batch_size
        self.beta = beta
        self.level = level
        self.num_betas = num_betas
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        with open(self.x[idx], 'rb') as f:
            full_im_ref, full_im_y =  _pickle.load(f)
            H,W = full_im_y.shape
            beta_i = np.zeros((H,W))
            
            if self.beta[idx] > 0.4:
                beta_i[:,:] = self.beta[idx]
            else:
                beta_i[:,:] = 0.4
        return full_im_ref.astype(np.float64), np.stack([full_im_y.astype(np.float64), 
                                                         beta_i.astype(np.float64)], axis=1)


class generator(Sequence):

    def __init__(self, x_set, batch_size=1):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        with open(self.x[idx], 'rb') as f:
            full_im_ref, full_im_y =  _pickle.load(f)

        return full_im_ref.astype(np.float32), full_im_y.astype(np.float32)
