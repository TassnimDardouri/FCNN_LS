import sys
import tensorflow as tf
sys.path.append('/path/to/FCNN_LS')
from utils.train_parser import Options
from utils.preprocess_data import U_input
from utils.helpers import get_args
from utils.model_utils import custom_loss

args  = get_args()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
opt = Options(trans = args.transform,
              method = args.method,
              level = args.level,
              dynamic = args.dynamic,
              num_neuron = args.num_neuron)

U_input(args, opt, Loss = custom_loss)