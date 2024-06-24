from tqdm import tqdm 
import numpy as np 
import torch
import pickle
import ukis_metrics.seg_metrics as segm
from model import SegmentationModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import os
torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_model(data_loader, model, device = 'cuda', norm = False, thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]):
    model = model.to(device)
    model.eval()
    
    tpfptnfn = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'n_valid_pixel': 0}

    images_dict, masks_dict, preds_dict = {}, {}, {}
    
    for threshold in thresholds:
        images_dict[threshold] = []
        masks_dict[threshold] = []
        preds_dict[threshold] = []

        with torch.no_grad():
            for images, masks in tqdm(data_loader):
                images = images.to(device, dtype = torch.float32)
                if norm:
                    images/= 255 # Normalize from 0...255 to 0...1
        
                logits = model(images, None)
                preds = torch.sigmoid(logits)
                preds = (preds > threshold).bool()
        
                masks = masks.bool().to(device)
        
                # Compute the classification performance
                tpfptnfn['tp'] += torch.sum((preds == 1) & (masks == 1)).cpu().numpy()
                tpfptnfn['fn'] += torch.sum((preds == 0) & (masks == 1)).cpu().numpy()
                tpfptnfn['tn'] += torch.sum((preds == 0) & (masks == 0)).cpu().numpy()
                tpfptnfn['fp'] += torch.sum((preds == 1) & (masks == 0)).cpu().numpy()
                tpfptnfn['n_valid_pixel'] += np.prod(masks.shape) # Get the number of pixels
                # Pick a random image to add to our examples
                if len(images_dict[threshold]) < 100:
                    random_idx = np.random.randint(images.shape[0])
                    images_dict[threshold].append(images.cpu().numpy()[random_idx])
                    masks_dict[threshold].append(masks.cpu().numpy()[random_idx])
                    preds_dict[threshold].append(preds.cpu().numpy()[random_idx])
            metrics = segm.segmentation_metrics(tpfptnfn)
            print(threshold, metrics)
        images_dict[threshold] = np.array(images_dict[threshold][:100])
        masks_dict[threshold] = np.array(masks_dict[threshold][:100])
        preds_dict[threshold] = np.array(preds_dict[threshold][:100])
        
    return images_dict, masks_dict, preds_dict

def get_model(model_ckpt, device = 'cuda'):
    # Take the same path and find the associated parameter file
    path = '/'.join(model_ckpt.split('/')[:-1]) + '/params.pkl'
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    print('Model parameters:')
    print(loaded_dict) # Show the parameter file's contents

    # Get the color channels that the model uses
    channels = loaded_dict['channels']
    
    # Load the model using the checkpoint path and chosen image channels (e.g. RGB)
    model = SegmentationModel('efficientnet-b0', 'imagenet', channels = channels)
    model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cuda')))
    model.eval()
    return model, channels
    
    return channels

def plot_predictions(images, masks, preds, title = ''):
    os.makedirs('results', exist_ok = True)
    for idx in range(images.shape[0]):
        # idx = random.randint(0, images.shape[0]-1) # 5
        # print(f'idx = {idx}')
        
        # Example image: has shape (1, 3, 512, 512)
        # We squeeze the first dimension -> (3, 512, 512)
        # Then transpose so the RGB channels are last -> (512, 512, 3)
        # And make sure the colors are integers in the range 0...255
        example_image = np.squeeze(images[idx])*255
        example_image = np.round(example_image).astype(int)
        example_image = example_image.transpose(1,2,0)
        
        # Also squeeze dimensions of the predictions and mask
        # By using a masked version, we make the imshow transparent if there is no water
        example_pred = np.squeeze(preds[idx]).astype(int)
        example_pred = np.ma.masked_where(example_pred == 0, example_pred)
        
        
        example_mask = np.squeeze(masks[idx]).astype(int)
        example_mask = np.ma.masked_where(example_mask == 0, example_mask)
        
        fig, axs = plt.subplots(2, 2)
        for ax_row in axs:
            for ax in ax_row:
                ax.imshow(example_image[:, :, :3])
                ax.axis('off')
        
        axs[0, 1].imshow(example_pred, alpha = 0.6, vmin = 0, vmax = 1, cmap = 'Greens')
        axs[1, 1].imshow(example_pred, alpha = 0.6, vmin = 0, vmax = 1, cmap = 'Greens')
        
        axs[1, 0].imshow(example_mask, alpha = 0.6, vmin = 0, vmax = 1, cmap = 'Reds')
        axs[1, 1].imshow(example_mask, alpha = 0.6, vmin = 0, vmax = 1, cmap = 'Reds')
        
        
        axs[0, 0].set_title("Image")
        axs[0, 1].set_title("Image + prediction")
        axs[1, 0].set_title("Image + mask")
        axs[1, 1].set_title("Image + prediction + mask")
    
        fig.suptitle(f'{title} IDX={idx}')
        
        plt.tight_layout()
    
        red_patch = mpatches.Patch(color='red', label='Mask')
        blue_patch = mpatches.Patch(color='green', label='Prediction')
        plt.legend(handles=[red_patch, blue_patch], loc="upper center", bbox_to_anchor=(1.5, 2.5))
        
        plt.savefig(f'./results/{title}_{idx}.png', bbox_inches='tight', transparent = True)
        plt.show()
        