import numpy as np
from skimage import io, metrics
from skimage import metrics
import glob
import _pickle

def w_L1(x,y, weight):
    return (1/weight)*np.mean(np.abs(x-y))

def rgb2gray(rgb):
    gray = np.empty((rgb.shape[0],rgb.shape[1]))
    if len(rgb.shape)==3:
        if rgb.shape[2] == 4:
            rgb = rgb[:,:,:-2]
        if rgb.shape[2] == 3:
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            gray = gray
        elif rgb.shape[2] == 2:
            r, g = rgb[:,:,0], rgb[:,:,1]
            gray = (0.299 + 0.114) * r + 0.587 * g
    else:
        gray = rgb
    return gray.astype('float64')


def load_images(im_path):
    images = []
    d = {}
    addrs = glob.glob(im_path)
    for i in range(len(addrs)):
        img = io.imread(addrs[i])
        if img is not None:
            images.append(img)
            d.update({i:addrs[i]})
    for i in range(len(images)):
        images[i] = rgb2gray(images[i])
    return images, d

def load_im_ordered(im_path):
    images = []
    d = {}
    addrs = glob.glob(im_path+ '/*.png')
    for i in range(len(addrs)):
        path = im_path+ '/image'+str(i+1)+'.png'
        print(path)
        img = io.imread(path)
        if img is not None:
            images.append(img)
            d.update({i:path})
    for i in range(len(images)):
        images[i] = rgb2gray(images[i])
    return images, d

def subband(image):
    x0 = image[::2,::2]
    x1 = image[::2,1::2]
    x2 = image[1::2,::2]
    x3 = image[1::2,1::2]
    return x0, x1, x2, x3


def Mse(y_test, out):
    return metrics.mean_squared_error(y_test, out)


def Psnr(y_test, out, Data_Range):
    return metrics.peak_signal_noise_ratio(y_test, out, data_range = Data_Range)

def Mae(x, y):
    return np.mean(abs(x-y))

def L_beta(x,y):
    loss = np.mean(abs(x-y)**0.75)
    return loss

def get_num_im(im_path):
    addrs = glob.glob(im_path)
    num_im = len(addrs)
    return num_im


def load_pickle(path, i):
    
    path_i = path + '/%d.pickle'%(i)
    with open(path_i,'rb') as f:
        argsout = _pickle.load(f)
    return argsout


def dump_pickle(path, i, *argsin):
    
    path_i = path + '/%d.pickle'%(i)
    
    if len(argsin) == 1:
        
        with open(path_i , 'wb') as f:
            _pickle.dump(argsin[0], f)
    else:
        
        with open(path_i , 'wb') as f:
            _pickle.dump(list(argsin), f)

def input_generator(n, im_path):
    number = 0
    count = 0
    len_data = get_num_im(im_path+'/*.pickle')
    while 1:
        batch_ref, batch_y = load_pickle(im_path, number)
        count+=1
        number+=1
        if number == len_data:
            number = 0
        if count % n == 0:
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