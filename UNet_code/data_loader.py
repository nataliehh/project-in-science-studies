import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import rasterio

class CustomDataLoader(Dataset):
    def __init__(self, image_path, mask_path, height = 512, width = 512, data_type = '', expand_dims = True, channels = 'r.g.b'):
        super().__init__()
        self.X = sorted(glob.glob(image_path))
        self.y = sorted(glob.glob(mask_path))
        assert len(self.X) > 0, f"Number of images (X) is zero. Is this path correct: {mask_path}"
        assert len(self.y) > 0, f"Number of masks (y) is zero. Is this path correct: {image_path}"
        self.len = len(self.X)
        self.data_type = data_type
        self.expand_dims = expand_dims
        self.transform = A.Compose([
            A.Resize(height,width),
            A.HorizontalFlip(),
        ])
        self.channels = channels.split('.')
        
    def __getitem__(self, idx):
        # Open with rasterio and shape into an RGB image
        img = rasterio.open(self.X[idx]).read()
        b, g, r, nir, swir1, swir2, slope  = img
        channel_dict = {'r': r, 'g': g, 'b': b, 'nir': nir, 'swir1': swir1, 'swir2': swir2}
        used_channels = []
        for channel in self.channels:
            if channel in channel_dict:
                used_channels.append(channel_dict[channel])
        img = np.stack(used_channels, axis = -1).astype(np.float64)
        # img = np.stack((r,g,b), axis = -1).astype(np.float64)

        # Get rid of the first, empty dimension
        mask = rasterio.open(self.y[idx]).read()[0, :, :]
        
        img, mask = np.array(img), np.array(mask)
        preprocess_fn = self.transform(image=img, mask=mask)
        img = preprocess_fn['image']
        mask = preprocess_fn['mask']

        if self.data_type == 'float':
            img = img.astype(np.float32)
            mask = mask.astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

        if self.expand_dims:
            mask = np.expand_dims(mask, axis=0)
        
        mask = torch.tensor(mask)

        return img, mask
    
    def __len__(self):
        return self.len
