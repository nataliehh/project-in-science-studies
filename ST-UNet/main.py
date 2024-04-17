import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
from torchviz import make_dot
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='S1S2', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
parser.add_argument('--device', type=str, default = 'cuda', help='which device to use for training')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.img_size = 256
    args.batch_size = 8
    dataset_name = args.dataset
    dataset_config = {
        'S1S2': {
            'root_path': '../data_prepped/{}/img/*',
            'mask_path': '../data_prepped/{}/msk/*',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.mask_path = dataset_config[dataset_name]['mask_path']
    # args.list_dir = None # dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False #True
    args.exp = 'STUNet_' + dataset_name + str(args.img_size)
    # Ensure we have the 'networks' folder
    os.makedirs('networks', exist_ok = True)

    # Get the current time and date
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Add specifications of model to checkpoint path, e.g. store the batch size, learning rate...
    snapshot_path = f"./networks/{args.exp}/{date_str}_{args.vit_name}_lr{args.base_lr}_bs{args.batch_size}_img{args.img_size}_skip{args.n_skip}"
    
    # Leave some parameters out if they're default
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path 
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path # Skip if default
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    
    print('-------------------------------------------')
    print(snapshot_path)
    print('------------------------------------------')

    os.makedirs(snapshot_path, exist_ok = True)
    
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()  #

    trainer = {dataset_name: trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)
