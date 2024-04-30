import torch
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from tqdm import tqdm
import numpy as np

def evaluate(model, dataloader, args):
    device = torch.device(args.device)
    # model.eval() # Set to evaluation mode (no backprop)

    # Track total number of samples and total loss
    num_samples = 0
    cumulative_loss = 0.0

    # Initialize losses
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    with torch.no_grad(): # Doubly ensure we aren't training
        for batch in tqdm(dataloader):
            image_batch, label_batch = batch
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # image_batch = image_batch.cpu()
            # image_batch = image_batch.numpy().astype(np.float32)

            # image_batch = torch.from_numpy(image_batch)
            # image_batch = image_batch.cuda()
            image_batch = image_batch.float()
            outputs = model(image_batch) # Predict
            # print('outputs', outputs)

            # Compute loss as average of crossentropy and dice loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice  # loss

            cumulative_loss += loss * args.batch_size # Sum up loss 'per sample'
            if torch.isnan(loss):
                print('loss is nan!', loss_ce, loss_dice)
            num_samples += args.batch_size
        
        avg_loss = cumulative_loss / num_samples # Average the loss over all samples
        print('cumulative loss:', cumulative_loss)
        print('num samples:', num_samples)
    model.train() # Set model to be in train mode again
    return avg_loss