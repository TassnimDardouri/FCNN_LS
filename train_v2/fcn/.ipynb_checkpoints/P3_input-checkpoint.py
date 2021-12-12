import sys
sys.path.append('/data-nfs/tasnim/code_v0')
from utils.train_parser import Options
from utils.preprocess_data import X3_input
from utils.helpers import get_args

args = get_args()
opt = Options(trans = args.transform, 
              method = args.method, 
              level = args.level, 
              dynamic = args.dynamic, )

X3_input(opt)