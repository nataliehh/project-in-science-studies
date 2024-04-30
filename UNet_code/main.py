# Adapted from: https://www.kaggle.com/code/uraninjo/satellite-water-bodies-pytorch-segmentation

import torch
from torch.utils.data import DataLoader

import os
import numpy as np
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import pickle

from model import SegmentationModel
from data_loader import CustomDataLoader
from train import train_fn, eval_fn
from utils import select_cpu_or_gpu


import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='max. epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float,  default=0.001, help='learning rate')
parser.add_argument('--encoder', type=str, default='efficientnet-b0', help = 'vision encoder architecture')
parser.add_argument('--weights', type=str, default='imagenet', help = 'pretrained weights')
parser.add_argument('--model_ckpt_path', type=str, default='./networks', help = 'where to store model checkpoints')
parser.add_argument('--data_path', type=str, default='../data_prepped', help = 'where dataset is stored')
parser.add_argument('--channels', type=str, default='r.g.b', help = 'which image channels (i.e. R, G, B, NIR, SWIR1, SWIR2) to use, separated by comma')
args = parser.parse_args()

if __name__ == "__main__":
    args.channels = args.channels.lower()
    device = select_cpu_or_gpu() 
    lr = args.lr
    
    image_path = args.data_path + '/{}/img/*'
    mask_path = args.data_path + '/{}/msk/*'
     
    # Use custom data loaders for S1S2 dataset
    train_dataset = CustomDataLoader(image_path.format('train'), mask_path.format('train'), channels = args.channels)
    valid_dataset = CustomDataLoader(image_path.format('val'), mask_path.format('val'), channels = args.channels)

    # Use standard pytorch data loader on the custom dataset
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 4)
    
    
    #################################################
    # UNCOMMENT TO GET CLASS BALANCE
    # positive_instances, negative_instances = 0, 0
    
    # for train_batch in tqdm(train_loader):
    #     image, mask = train_batch
    #     positive_instances += torch.sum(mask).item()
    #     negative_instances += torch.sum(1-mask).item()
    
    # total_instances = positive_instances + negative_instances
    # positive_ratio = positive_instances/total_instances
    # negative_ratio = 1 - positive_ratio
    # print(f'pos ratio: {positive_ratio}\tneg ratio: {negative_ratio}')
    #################################################
    
    model = SegmentationModel(args.encoder, args.weights, channels = args.channels)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas = (0.9, 0.9), weight_decay = 0.01)
    
    # Visualization - won't work in Python script
    # img, mask = train_dataset[4]
    
    # f, axarr = plt.subplots(1,2) 
    # axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
    # axarr[0].imshow(np.transpose(img.numpy(), (1,2,0)))
    
    best_valid_lovasz_loss = np.Inf
    best_valid_bce_loss = np.Inf
    best_valid_avg_loss = np.Inf
    
    # Get the current time and date
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args_str = f'{date_str}_encoder_{args.encoder}_weights_{args.weights}_epochs_{args.epochs}_batchsize_{args.batch_size}_lr_{args.lr}'
    logging.info(str(args))
    
    os.makedirs(args.model_ckpt_path, exist_ok = True)
    snapshot_path = f'{args.model_ckpt_path}/{args_str}'
    os.makedirs(snapshot_path, exist_ok = True)

    # Store the parameters as a dictionary, so we can retrieve them
    args_dict = vars(args)
    with open(snapshot_path + '/params.pkl', 'wb') as f:
        pickle.dump(args_dict,f)
    
    print('log file:', snapshot_path + "/log.txt")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S') 
    
    logging.info('Begin training.')
    epochs_without_improvement = 0
    
    for i in range(args.epochs):
        
        train_loss = train_fn(train_loader, model, optimizer, device)
        valid_loss = eval_fn(valid_loader, model, device)
        
        train_lovasz, train_bce = train_loss
        valid_lovasz, valid_bce = valid_loss
    
        # Defining average loss - not using this yet!!!
        train_avg_loss = train_lovasz * .5 + train_bce * .5
        valid_avg_loss = valid_lovasz * .5 + valid_bce * .5 
        
        results = f'Epochs:{i+1}\nTrain_loss -->\tLovasz: {train_lovasz:.5f}\tBCE: {train_bce:.5f}\tAvg {train_avg_loss:.5f} \
        \nValid_loss -->\tLovasz: {valid_lovasz:.5f}\tBCE: {valid_bce:.5f}\tAvg {valid_avg_loss:.5f}'
        print(results)
        logging.info(results)
        if valid_avg_loss < best_valid_avg_loss: #or valid_bce < best_valid_bce_loss
            # Save the best model
            path = "{}/{}/model_{:.5f}_val_avg.pt".format(args.model_ckpt_path, args_str, valid_avg_loss)
            torch.save(model.state_dict(), path)
            print('Model Saved')
            logging.info(f'Model saved at {path}')
            
            # Update what the best loss is so far
            best_valid_avg_loss = valid_avg_loss
            best_valid_lovasz_loss = valid_lovasz
            best_valid_bce_loss = valid_bce
            epochs_without_improvement = 0 # Reset counter
        elif valid_avg_loss > best_valid_avg_loss:
            epochs_without_improvement += 1
            # If we don't improve for 3 consecutive epochs, reduce learning rate by a factor of 0.1 (i.e. scale by 0.9?)
            if epochs_without_improvement % 3 == 0:
                logging.info(f'No improvement for 3 consecutive epochs. Rescaling learning rate from {lr} to {lr*0.9}.')
                lr *= 0.9 
                # Update the learning rate for the optimizer
                for group in optimizer.param_groups:
                    group['lr'] = lr
    
            # Apply early stopping if we don't improve for 9 consecutive 
            if epochs_without_improvement == 9:
                logging.info('No improvement for 9 consecutive epochs. Applying early stopping.')
                break
        else: 
            epochs_without_improvement = 0