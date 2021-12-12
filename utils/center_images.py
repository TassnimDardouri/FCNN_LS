import numpy as np
from multiprocessing import Pool
import os

def center_image(file_path):
    print(file_path)
    image = np.load(file_path)
    image = image - np.mean(image) 
    save_path = file_path.replace('ordered', 'centered_ordered')
    print(save_path)
    np.save(save_path, image)
    
if __name__ == '__main__':
    
    pool = Pool(30)
    dir_train = '/path/to/datasets/clic/ordered_clic_train_dataset/'
    dir_val = '/path/to/datasets/clic/ordered_clic_valid_dataset/'
    dir_test = '/path/to/datasets/clic/ordered_clic_test_dataset/'
    
    train_path_list = [os.path.join(dir_train, file) for file in os.listdir(dir_train) \
                       if 'npy' in file]
    val_path_list = [os.path.join(dir_val, file) for file in os.listdir(dir_val) \
                    if 'npy' in file]
    test_path_list = [os.path.join(dir_test, file) for file in os.listdir(dir_test) \
                     if 'npy' in file]
    
    all_im_list = train_path_list + val_path_list + test_path_list
    pool.map(center_image, all_im_list)
    pool.close()