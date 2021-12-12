import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import load_model, Model
import os

import sys
sys.path.append('/data/tdardour_data/image_comp/codes')
from utils.train_parser import Options
from utils.dataloader import full_loader, input_generator, get_data_length, generator
from utils.data_utils import load_pickle
from models.fcn import build_model
from utils.preprocess_data import create_directory
from utils.helpers import get_args
from utils.model_utils import custom_loss

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

x_train , y_train = load_pickle(opt.P2_train_path, 1)
input_dim = x_train.shape[1]
model = load_model('/data/tdardour_data/image_comp/weights/models_16n/' + \
                   'fcn_42_AG_16n/level%s_X2_fcn_42_AG_16n.h5'%(args.level))
model = Model(inputs=model.input, outputs=model.layers[-2].output)
model.summary()
adam = optimizers.Adam(lr=opt.lr, 
                       beta_1=0.9, 
                       beta_2=0.999, 
                       amsgrad=False, 
                       decay=opt.decay)

model.compile(loss=custom_loss, optimizer=adam)

terminate_onnan = callbacks.TerminateOnNaN()
csv_logger = callbacks.CSVLogger(opt.P2_log_path)

filepath = opt.P2_model_path
create_directory(os.path.dirname(filepath))

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                      verbose=1, save_best_only=True, 
                                      save_weights_only=False, mode='auto')

#b_size = 65536
len_train = get_data_length(opt.P2_train_path, batch_size=None)
len_test = get_data_length(opt.P2_test_path, batch_size=None)

train_dir = opt.P2_train_path
test_dir = opt.P2_test_path
print(train_dir, test_dir)

train_list = [os.path.join(train_dir, file) for \
              file in os.listdir(train_dir) if 'pickle' in file]

test_list = [os.path.join(test_dir, file) for \
              file in os.listdir(test_dir) if 'pickle' in file]

train_generator = generator(x_set = train_list)
test_generator = generator(x_set = test_list)

model.fit(train_generator,
          batch_size=1,
          steps_per_epoch=len_train,
          epochs=opt.epochs,
          callbacks=[terminate_onnan, checkpoint, csv_logger],
          validation_data=test_generator,
          validation_batch_size=1,
          validation_steps=len_test,
          workers=opt.num_workers,
          use_multiprocessing=False,
          shuffle=True)