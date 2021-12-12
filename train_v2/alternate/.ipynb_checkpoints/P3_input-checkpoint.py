import sys
sys.path.append('/data/tdardour_data/image_comp/codes')
from utils.train_parser import Options
from utils.preprocess_data import X3_input
from utils.helpers import get_args

args = get_args()
opt = Options(trans = args.transform,
              method = args.method,
              level = args.level,
              dynamic = args.dynamic,
              num_neuron = args.num_neuron)

X3_input(opt)