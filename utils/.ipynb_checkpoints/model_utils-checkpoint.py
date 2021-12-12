import numpy as np
import tensorflow as tf
import scipy.linalg as slin
import tensorflow.keras
import tensorflow_probability as tfp
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, PReLU
from tensorflow.keras import optimizers, callbacks
import _pickle
import glob
import sys

from utils.data_utils import input_generator, input_generator_beta, get_num_im, Mse, Psnr, load_pickle
from utils.dataloader import get_data_length
from utils.logger import Log

def round_mse(y_true, y_pred):
    y_pred_rounded_NOT_differentiable = tf.round(y_pred)
    y_pred_rounded_differentiable = (y_pred - (tf.stop_gradient(y_pred) - y_pred_rounded_NOT_differentiable))
    loss = k.mean(tf.math.pow((y_true - y_pred_rounded_differentiable),2))
    return loss

def L_3_4(y_true, y_pred):
    beta = tf.constant(0.75)
    loss = k.mean(tf.math.pow(tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))
    return loss

def L_06(y_true, y_pred):
    beta = tf.constant(0.6)
    loss = k.mean(tf.math.pow(tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))
    return loss

def L_const_beta(beta):
    def Loss(y_true, y_pred):
        loss = k.mean(tf.math.pow(
            tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))
        return loss
    #return a function
    return Loss

def log_L_const_beta(beta=1):
    def Loss(y_true, y_pred):
        loss = (1/beta)*tf.math.log(k.mean(tf.math.pow(
            tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))+ 0.00002)
        return loss
    #return a function
    return Loss

def L_beta_plus_Log_L_beta(alpha, beta):
    def Loss(y_true, y_pred):
        loss = (1/beta)*1.43*tf.math.log(k.mean(tf.math.pow(
                tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))+ 0.00002) +\
                ((1/alpha)**beta)*k.mean(tf.math.pow(
                tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))
        return loss
    return Loss

def L_beta_plus_Log_L_beta_adapt(y_true, y_pred):
    beta = tf.cast(y_true[:,1][0], tf.float64)
    y_true = tf.cast(tf.expand_dims(y_true[:,0], axis=1), tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    loss = (1/beta)*1.43*tf.math.log(k.mean(tf.math.pow(
            tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))+ 0.02) +\
            k.mean(tf.math.pow(tf.math.pow((y_true - y_pred + 0.02),2), beta/2))
    return loss

def Log_L_beta_adapt(y_true, y_pred):
    beta = tf.cast(y_true[:,1][0], tf.float64)
    y_true = tf.cast(tf.expand_dims(y_true[:,0], axis=1), tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    loss = (1/beta)*tf.math.log(k.mean(tf.math.pow(
            tf.math.pow((y_true - y_pred  + 0.02),2), beta/2))+ 0.02)
    return loss

def L_beta(y_true, y_pred):
    const = tf.constant(2, dtype = tf.float32)
    beta = tf.expand_dims(y_true[:,1], axis=1)
    loss = k.mean(tf.math.pow((tf.math.pow((tf.expand_dims(y_true[:,0], axis=1) - y_pred),const) + 0.02), beta/2))
    return loss

def custom_loss(y_true,y_pred):
    Rupdx = k.dot(k.transpose(y_pred), y_pred)
    r = k.sum(tf.math.multiply(y_pred, y_true), axis = 0)
    rjourx = k.expand_dims(r,axis=-1)
    p = k.dot((tfp.math.pinv(Rupdx)), rjourx)
    pred = k.sum(tf.math.multiply(y_pred,k.transpose(p)), axis =1)
    pred = k.expand_dims(pred,axis=-1)
    loss = k.mean(k.square(y_true - pred))
    return loss

loss_dict = {L_06: 'L_06',
             L_3_4: 'L_3_4', 
             L_beta: 'L_beta',
             custom_loss:'custom_loss'}

def adaptive_predict(model, ref, target, p):
    out = model.predict(ref, batch_size = ref.shape[0])
    out = out.astype('float64')
    if np.isscalar(p):
        R = np.matmul(np.transpose(out), out)
        r = np.expand_dims(np.sum(np.multiply(out,target), axis = 0),axis=1)
        if slin.det(R) == 0:
            p = np.matmul((slin.pinv(R)),r)
        else:
            p = np.matmul((slin.inv(R)),r)
        p = p.reshape(1,p.shape[0])
    im_pred_flat = np.expand_dims(np.sum(np.multiply(out,p),axis=1),axis=1)
    return im_pred_flat, p

def import_models(path_p1, path_p2, path_p3, path_u, Loss, Compile):
    
    if type(Loss)==tuple:
        Loss_0 = Loss[0]
        Loss_1 = Loss[1]
        Loss_2 = Loss[2]

        model_p1 = load_model(path_p1, custom_objects = {'Loss_0': Loss_0})
        model_p2 = load_model(path_p2, custom_objects = {'Loss_1': Loss_1})
        model_p3 = load_model(path_p3, custom_objects = {'Loss_2': Loss_2})
        model_u = load_model(path_u)
        
    else:  
        
        if Compile == False:
            model_p1 = load_model(path_p1, compile = False)
            model_p2 = load_model(path_p2, compile = False)
            model_p3 = load_model(path_p3, compile = False)
            model_u = load_model(path_u, compile = False)

        elif Loss is not None:
            model_p1 = load_model(path_p1, custom_objects = {loss_dict[Loss]: Loss})
            model_p2 = load_model(path_p2, custom_objects = {loss_dict[Loss]: Loss})
            model_p3 = load_model(path_p3, custom_objects = {loss_dict[Loss]: Loss})
            model_u = load_model(path_u, custom_objects = {loss_dict[Loss]: Loss})
        else:
            model_p1 = load_model(path_p1)
            model_p2 = load_model(path_p2)
            model_p3 = load_model(path_p3)
            model_u = load_model(path_u)
    return model_p1, model_p2, model_p3, model_u


def Load_Models(filepath, num_levels, Loss, Compile = False):
    models = ()
    for i in range(num_levels):
        path_p3 = glob.glob(filepath + '/level' + str(i+1) + '_X3_' + '*.h5')
        path_p2 = glob.glob(filepath + '/level' + str(i+1) + '_X2_' + '*.h5')
        path_p1 = glob.glob(filepath + '/level' + str(i+1) + '_X1_' + '*.h5')
        path_u = glob.glob(filepath + '/level' + str(i+1) + '_U_' + '*.h5')
        print(path_p1[0], path_p2[0], path_p3[0], path_u[0])
        if type(Loss)==tuple:
            model_p1, model_p2, model_p3, model_u = import_models(path_p1[0],
            path_p2[0],
            path_p3[0],
            path_u[0],
            (Loss[i*3], Loss[(i*3)+1], Loss[(i*3)+2]), Compile)
        else:
            model_p1, model_p2, model_p3, model_u = import_models(path_p1[0], 
                                                                  path_p2[0], 
                                                                  path_p3[0], 
                                                                  path_u[0], 
                                                                  Loss, 
                                                                  Compile)
        models = models + (model_p1, model_p2, model_p3, model_u,)
    return models


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def clip(x):
    max_value = _to_tensor(255., x.dtype.base_dtype)
    zeros = _to_tensor(-np.inf ,x.dtype.base_dtype)
    x = tf.clip_by_value(x, zeros, max_value)
    return x

def train(model, train_path, test_path, csv_path, model_path, num_epoch, save_period):
    
    len_train = get_num_im(train_path + '/*.pickle')
    len_test = get_num_im(test_path + '/*.pickle')
    
    terminate_onnan = callbacks.TerminateOnNaN()
    csv_logger = callbacks.CSVLogger(csv_path)
    filepath = model_path
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=0, save_best_only=False, 
                                          save_weights_only=False, mode='auto', period = save_period)
    b_size = 1
    model.fit(input_generator(1,train_path), 
              batch_size=b_size,
              steps_per_epoch=len_train/b_size, 
              epochs=num_epoch, 
              verbose=1, 
              callbacks=[terminate_onnan, checkpoint, csv_logger],
              validation_data=input_generator(1,test_path), 
              validation_batch_size=b_size,
              validation_steps=len_test/b_size, 
              class_weight=None, 
              max_queue_size=50, 
              workers=1, 
              use_multiprocessing=False, 
              shuffle=True)
    
def train_beta(model, train_path, test_path, csv_path, model_path, num_epoch, save_period, beta_test, beta_train):
    
    len_train = get_num_im(train_path + '/*.pickle')
    len_test = get_num_im(test_path + '/*.pickle')
    
    terminate_onnan = callbacks.TerminateOnNaN()
    csv_logger = callbacks.CSVLogger(csv_path)
    filepath = model_path
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=0, save_best_only=False, 
                                          save_weights_only=False, mode='auto', period = save_period)
    
    model.fit(input_generator_beta(1, train_path, beta_train), 
                        batch_size=1, 
                        steps_per_epoch=len_train,
                        epochs=num_epoch, 
                        verbose=1, 
                        callbacks=[terminate_onnan, checkpoint, csv_logger],
                        validation_data=input_generator_beta(1,test_path, beta_test),
                        validation_batch_size=1,
                        validation_steps = len_test,
                        max_queue_size=50, 
                        workers=1, 
                        use_multiprocessing=False, 
                        shuffle=True, 
                        initial_epoch=0)
        
def get_opt_w(x,y):
    Rupdx = np.matmul(np.transpose(x), x)
    rjourx = np.expand_dims(np.sum(np.multiply(x,y), axis = 0),axis=1)
    if slin.det(Rupdx) == 0:
        p = np.matmul((slin.pinv(Rupdx)),rjourx)
    else:
        p = np.matmul((slin.inv(Rupdx)),rjourx)
    return p

def train_alternate(model, 
                    train_path, 
                    test_path, 
                    log_path, 
                    model_path, 
                    num_epoch, 
                    save_period=20, 
                    iters=5, 
                    train_batch_size = None,
                    val_batch_size = None):
    log = Log(log_path)
    get_theta_output = k.function([model.layers[0].input],
                                  [model.layers[-2].output])
    
    len_train = get_data_length(train_path, batch_size = train_batch_size)
    len_test = get_data_length(test_path, batch_size = val_batch_size)
    len_train_im = get_data_length(train_path, batch_size = None)
    len_test_im = get_data_length(test_path, batch_size = None)
    
    bias = np.zeros((1))
    hist_loss = np.zeros((num_epoch,1))
    val_mse = np.zeros((num_epoch,1))
    val_psnr = np.zeros((num_epoch,1))

    for epoch in range(num_epoch):

        hist = 0
        for image in range(1, len_train_im+1):
            full_ref_train, full_y_train = load_pickle(train_path, image)
            if train_batch_size == None:
                ref_train, y_train = full_ref_train, full_y_train
                for iter_ in range(iters):
                    List = []
                    out_train = get_theta_output([ref_train])[0].astype('float64')
                    p = get_opt_w(out_train,y_train)
                    List.append(p)
                    List.append(bias)
                    model.layers[-1].set_weights(List)
                    history = model.fit(ref_train, y_train, epochs=1, 
                                        batch_size = ref_train.shape[0], verbose=0)
                    if iter_ == iters - 1:
                        hist = hist + history.history['loss'][0]
                    sys.stdout.write('\r image ' + str(image) + ', iter' + str(iter_) + \
                                         ', processed in epoch' + str(epoch) + ', loss = '+ \
                                         str(history.history['loss'][0]))
                    sys.stdout.flush()
            else:
                
                for i in range(0, full_ref_train.shape[0], train_batch_size):
                    if i <= full_ref_train.shape[0]:
                        if i+train_batch_size <= full_ref_train.shape[0]:
                            ref_train = full_ref_train[i:i+train_batch_size,:]
                            y_train = full_y_train[i:i+train_batch_size,:]
                        else:
                            ref_train = full_ref_train[i:,:] 
                            y_train = full_y_train[i:,:]

                    for iter_ in range(iters):
                        List = []
                        out_train = get_theta_output([ref_train])[0].astype('float64')
                        p = get_opt_w(out_train,y_train)
                        List.append(p)
                        List.append(bias)
                        model.layers[-1].set_weights(List)
                        history = model.fit(ref_train, y_train, epochs=1, batch_size = ref_train.shape[0], verbose=0)
                        if iter_ == iters - 1:
                            hist = hist + history.history['loss'][0]
                        sys.stdout.write('\r image ' + str(image) + ', iter' + str(iter_) + \
                                         ', processed in epoch' + str(epoch) + ', loss = '+ \
                                         str(history.history['loss'][0]))
                        sys.stdout.flush()
                    
        hist_loss[epoch,:]= hist/(len_train)
        log.log_training(hist/(len_train), epoch)
        print('\n epoch: ' + str(epoch) + ', loss: ' + str(hist/(len_train)))
        
        mse = 0
        psnr = 0
        for i in range(1, len_test_im+1):

            ref_test, y_test = load_pickle(test_path, i)
            out_test = get_theta_output([ref_test])[0].astype('float64')
            p = get_opt_w(out_test, y_test)
            p = p.reshape(1,p.shape[0])
            pred = np.expand_dims(np.sum(np.multiply(out_test,p),axis=1),axis=1)
            mse = mse + Mse(y_test, pred)
            psnr = psnr + Psnr(y_test, pred, 255)
            sys.stdout.write('\r image ' + str(i) + ', processed in epoch' + \
                             str(epoch) + ', loss = '+ str(mse))
            sys.stdout.flush()

        val_mse[epoch,:] = mse/len_test
        val_psnr[epoch,:] = psnr/len_test
        log.log_validation(mse/len_test, epoch)
        print('\n epoch: ' + str(epoch) + ', loss: ' + str(mse/len_test))
        
        #with open(csv_path, 'wb') as file_pi:
        #    _pickle.dump([hist_loss, val_mse, val_psnr], file_pi)

        if (epoch +1) % save_period == 0:   
            trunc_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            trunc_model.save(model_path+'_%d.h5'%(epoch+1))
        
    log.close()
def build_model(in_dim, a0, a1, a2):
    model = Sequential()
    model.add(Dense(a0, kernel_initializer='normal', input_dim = in_dim))
    model.add(PReLU())
    model.add(Dense(a1, kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dense(a2, kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    return model