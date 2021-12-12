import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import os

import sys
sys.path.append('/data-nfs/tasnim/code_v0')
from utils.train_parser import Options
from utils.dataloader import full_loader, input_generator, get_data_length, generator
from utils.data_utils import load_pickle
from models.fcn import build_model, build_model_L2_regular, build_model_L1_regular
from utils.preprocess_data import create_directory
from utils.helpers import get_args
from utils.model_utils import L_beta, L_06, L_const_beta, round_mse

args  = get_args()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
opt = Options(trans = args.transform, method = args.method, 
              level = args.level, dynamic = args.dynamic, epochs = args.epochs, 
              lr = args.lr, decay = args.decay, num_neuron = args.num_neuron)

x_train , y_train = load_pickle(opt.U_train_path, 1)
input_dim = x_train.shape[1]
if args.add_penalty == 'L2':
    model = build_model_L2_regular(input_dim, args)
elif args.add_penalty == 'L1':
    model = build_model_L1_regular(input_dim, args)
else:
    model = build_model(input_dim, args)
model.summary()
    
adam = optimizers.Adam(lr=opt.lr, 
                       beta_1=0.9, 
                       beta_2=0.999, 
                       amsgrad=False, 
                       decay=opt.decay)

if args.Loss == 'round_mse':
    model.compile(loss=round_mse, optimizer=adam)
else:
    model.compile(loss=args.Loss, optimizer=adam)

terminate_onnan = callbacks.TerminateOnNaN()
csv_logger = callbacks.CSVLogger(opt.U_log_path)

filepath = opt.U_model_path
create_directory(os.path.dirname(filepath))

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                      verbose=1, save_best_only=True, 
                                      save_weights_only=False, mode='auto')

#b_size = 65536
len_train = get_data_length(opt.U_train_path, batch_size=None)
len_test = get_data_length(opt.U_test_path, batch_size=None)

train_dir = opt.U_train_path
test_dir = opt.U_test_path
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
          epochs=args.epochs, 
          callbacks=[terminate_onnan, checkpoint, csv_logger],
          validation_data=test_generator,
          validation_batch_size=1, 
          validation_steps=len_test,
          workers=opt.num_workers, 
          use_multiprocessing=False, 
          shuffle=True)