import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as k
import os
import numpy as np 
import _pickle
import math
import time

import sys
sys.path.append('/path/to/FCNN_LS')
from utils.train_parser import Options
from utils.dataloader import full_loader, get_data_length
from utils.data_utils import Mse
from utils.model_utils import get_opt_w
from utils.preprocess_data import create_directory
from utils.helpers import get_args
from utils.logger import Log

args  = get_args()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
opt = Options(trans = args.transform, method = args.method, 
              level = args.level, dynamic = args.dynamic, epochs = args.epochs, 
              lr = args.lr, decay = args.decay, num_neuron = args.num_neuron)

model = load_model('/path/toS/weights/models_'+str(args.num_neuron)+'n/' + \
                   'fcn_42_AG_'+str(args.num_neuron)+'n/level'+str(args.level)+\
                   '_U_fcn_42_AG_'+str(args.num_neuron)+'n.h5')
model.summary()
adam = optimizers.Adam(lr=opt.lr, 
                       beta_1=0.9, 
                       beta_2=0.999, 
                       amsgrad=False, 
                       decay=opt.decay)

model.layers[-1].trainable = False
model.compile(loss='mean_squared_error', optimizer=adam)

model_path = opt.U_model_path
create_directory(os.path.dirname(model_path))

len_train = get_data_length(opt.U_train_path, batch_size=None)
len_test = get_data_length(opt.U_test_path, batch_size=None)

train_dir = opt.U_train_path
test_dir = opt.U_test_path
print(train_dir, test_dir)

train_list = [os.path.join(train_dir, file) for \
              file in os.listdir(train_dir) if 'pickle' in file]

test_list = [os.path.join(test_dir, file) for \
              file in os.listdir(test_dir) if 'pickle' in file]

train_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
test_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

iters=5

log = Log(opt.U_log_path)
get_theta_output = k.function([model.layers[0].input],
                              [model.layers[-2].output])

bias = np.zeros((1))
hist_loss = np.zeros((opt.epochs,1))
val_mse = np.zeros((opt.epochs,1))

val_loss = math.inf
for epoch in range(opt.epochs):
    epoch_start = time.time()
    hist = 0
    for image_path in train_list:
        with open(image_path, 'rb') as f:
            ref_train, y_train = _pickle.load(f)
        for iter_ in range(iters):
            List = []
            out_train = get_theta_output([ref_train])[0].astype('float64')
            p = get_opt_w(out_train,y_train)
            List.append(p)
            List.append(bias)
            model.layers[-1].set_weights(List)
            loss = model.train_on_batch(ref_train, y_train, 
                           sample_weight=None, 
                           class_weight=None, 
                           reset_metrics=True,
                           return_dict=False)

            if iter_ == iters - 1:
                hist = hist + loss
            sys.stdout.write('\r image ' + str(os.path.basename(image_path)) \
                             + ', iter' + str(iter_) + ', processed in epoch' \
                             + str(epoch) + ', loss = '+ str(loss))
            sys.stdout.flush()

    hist_loss[epoch,:]= hist/(len_train)
    print('\n epoch: ' + str(epoch) + ', loss: ' + str(hist/(len_train)))

    mse = 0
    for image_path in test_list:
        with open(image_path, 'rb') as f:
            ref_test, y_test = _pickle.load(f)
        out_test = get_theta_output([ref_test])[0].astype('float64')
        p = get_opt_w(out_test, y_test)
        p = p.reshape(1,p.shape[0])
        pred = np.expand_dims(np.sum(np.multiply(out_test,p),axis=1),axis=1)
        mse = mse + Mse(y_test, pred)
        sys.stdout.write('\r image ' + str(os.path.basename(image_path)) + ', processed in epoch' + \
                         str(epoch) + ', loss = '+ str(mse))
        sys.stdout.flush()

    val_mse[epoch,:] = mse/len_test
    log.log_epoch_end(epoch, hist/len_train, mse/len_test)
    print('\n epoch: ' + str(epoch) + ', loss: ' + str(mse/len_test))
    epoch_end = time.time()
    print(str(epoch_end-epoch_start)+ ' sec/epoch')
    if mse/len_test < val_loss:   
        trunc_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        trunc_model.save(model_path)
        print('val_loss improved from:', val_loss,'to',mse/len_test,', model saved to: ', model_path)
        val_loss = mse/len_test
log.close()