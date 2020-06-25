"""main.py"""

import argparse
import numpy as np
import pprint
import torch
import os

from gpu_utils import auto_select_gpu

os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
from config import get_params

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


from solver import Solver


parser = argparse.ArgumentParser(description='toy Beta-VAE')

parser.add_argument('--train', default=True, help='train or traverse')
parser.add_argument('--seed', default=1, type=int, help='random seed')

parser.add_argument('--max_iter', default=1e5, type=int, help='maximum training iteration')

parser.add_argument('--viz_on', default=True, help='enable visdom visualization')
parser.add_argument('--save_output', default=True, help='save traverse images and gif')

parser.add_argument('--gather_step', default=2000, type=int, help='numer of iterations after which data is gathered for visdom')
parser.add_argument('--display_step', default=2000, type=int, help='number of iterations after which loss data is printed and visdom is updated')

parser.add_argument('--save_step', default=4000, type=int, help='number of iterations after which a checkpoint is saved')


parser.add_argument('--beta', dest='beta',
                    default=250, type=float)
parser.add_argument('--checkpoint_file', dest='checkpoint_file',
                    default=None, type=str)
# data
parser.add_argument('--dataset', dest='dataset',
                    help='Training dataset',
                    default='casia', type=str)

# checkpoint
parser.add_argument('--output_dir', dest='output_dir',
                    help='the dir save result',
                    default='output1', type=str)
parser.add_argument('--comment', type=str, default='_alex_beta250')
args = parser.parse_args()


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    params = get_params(args.dataset)
    params.update(vars(args))
    pprint.pprint(params)

    net = Solver(params)

    if args.train:
        net.train()
    else:
        net.viz_traverse(0)


'''

python main.py --dataset celeba --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 10 --viz_name celeba_H_beta10_z10
'''

if __name__ == "__main__":
    main(args)
