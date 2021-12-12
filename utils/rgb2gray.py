from PIL import Image
import numpy as np
from multiprocessing import Pool
import os
from data_utils import rgb2gray

def convert_rgb_gray(file_path):
    print(file_path)
    image = Image.open(file_path).convert('L')
    save_path = file_path.replace('ordered', 'ordered_gray')
    print(save_path)
    image.save(save_path)
    
if __name__ == '__main__':
    
    pool = Pool(30)
    dir_train = '/path/to/datasets/clic/ordered_clic_train_dataset/'
    dir_val = '/path/to/datasets/clic/ordered_clic_valid_dataset/'
    dir_test = '/path/to/datasets/clic/ordered_clic_test_dataset/'
    
    train_path_list = [os.path.join(dir_train, file) for file in os.listdir(dir_train) \
                       if 'png' in file]
    val_path_list = [os.path.join(dir_val, file) for file in os.listdir(dir_val) \
                    if 'png' in file]
    test_path_list = [os.path.join(dir_test, file) for file in os.listdir(dir_test) \
                     if 'png' in file]
    
    all_im_list = train_path_list + val_path_list + test_path_list
    pool.map(convert_rgb_gray, all_im_list)
    pool.close()