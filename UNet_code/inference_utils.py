from tqdm import tqdm 
import numpy as np 
import torch
import pickle
import ukis_metrics.seg_metrics as segm
from model import SegmentationModel
torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_model(data_loader, model, device = 'cuda', thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]):
    model = model.to(device)
    
    tpfptnfn = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'n_valid_pixel': 0}
    
    for threshold in thresholds:
        with torch.no_grad():
            for images, masks in tqdm(data_loader):
                images = images.to(device, dtype = torch.float32)
        
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
        
            metrics = segm.segmentation_metrics(tpfptnfn)
            print(threshold, metrics)

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
    