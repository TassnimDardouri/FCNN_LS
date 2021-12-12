import argparse

def get_args():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--level', type=int, default=1, help='')
    parser.add_argument('--method', type=str, default='fcn_42_AG', help='')
    parser.add_argument('--dynamic', type=int, default=0, help='')
    parser.add_argument('--transform', type=str, default='42_AG', help='')
    parser.add_argument('--epochs', type=int, default=200, help='')
    parser.add_argument('--lr', type=float, default=1e-2, help='')
    parser.add_argument('--decay', type=float, default=1e-4, help='')
    parser.add_argument('--val_path', type=str, default='/path/to/datasets/clic/ordered_clic_test_dataset', 
                        help='')
    parser.add_argument('--Loss', type=str, help='mean_squared_error, mean_absolute_error, L_beta')#default='mean_squared_error',
    parser.add_argument('--beta', type=str, default='0.6', help='0.62,0.62,0.72,0.57,0.57,0.63,0.54,0.54,0.56')
    parser.add_argument('--num_neuron', type=int, default=16, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--add_penalty', type=str, default=None, help='L1, L2')
    args = parser.parse_args()
    return args