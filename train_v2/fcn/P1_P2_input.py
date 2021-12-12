import tensorflow as tf
import sys
sys.path.append('/path/to/FCNN_LS')
from utils.train_parser import Options
from utils.preprocess_data import X1_X2_input
from utils.helpers import get_args
from utils.model_utils import L_beta, L_06, L_const_beta, log_L_const_beta, round_mse

args = get_args()
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
        
opt = Options(trans = args.transform, 
              method = args.method, 
              level = args.level, 
              dynamic = args.dynamic)

if args.Loss == 'L_beta':
    loss = L_beta
elif args.Loss == 'round_mse':
    loss = round_mse
elif args.Loss == 'L_06':
    loss = L_06
elif args.Loss == 'log_L_beta':
    loss=log_L_const_beta
elif args.Loss == 'log_L1':
    loss=log_L_const_beta
elif args.Loss == 'log_L2':
    loss=log_L_const_beta
else:
    loss = None

X1_X2_input(args, opt, Loss = loss)
        