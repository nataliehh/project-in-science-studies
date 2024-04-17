import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train_fn(data_loader, model, optimizer, device):
    model.train()
    total_lovaszloss=0.0
    total_bceloss=0.0
    for images, masks in tqdm(data_loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        logits, lovaszloss, bceloss = model(images, masks)
        lovaszloss.backward(retain_graph = True)
        bceloss.backward()
        optimizer.step()
        total_lovaszloss += lovaszloss.item()
        total_bceloss += bceloss.item()  

    return total_lovaszloss/len(data_loader), total_bceloss/len(data_loader)

def eval_fn(data_loader, model, device, sample_num = 2, ratio = 0.05, visualization = False):
    model.eval()
    total_lovaszloss = 0.0
    total_bceloss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device, dtype = torch.float32)
            masks = masks.to(device, dtype = torch.float32)

            logits, lovaszloss, bceloss = model(images,masks)
            total_lovaszloss += lovaszloss.item()
            total_bceloss += bceloss.item()
            
        #Visualization
        if visualization:
            for i in range(1):
                image, mask = next(iter(data_loader))
                image = image[sample_num]
                mask = mask[sample_num]
                logits_mask = model(image.to('cuda', dtype = torch.float32).unsqueeze(0))
                pred_mask = torch.sigmoid(logits_mask)
                pred_mask = (pred_mask > ratio)*1.0
                f, axarr = plt.subplots(1,3) 
                axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
                axarr[0].imshow(np.transpose(image.numpy(), (1,2,0)))
                axarr[2].imshow(np.transpose(pred_mask.detach().cpu().squeeze(0), (1,2,0)))
                plt.show()
            
    return total_lovaszloss/len(data_loader),total_bceloss/len(data_loader)
