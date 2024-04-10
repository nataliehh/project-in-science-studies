import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import lovasz_softmax, DiceLoss
from evaluate import evaluate

sys.path.append('../UNet_code')
from data_loader import CustomDataLoader

def cal(pred, label, num_categories = 5):
    tp = np.zeros(num_categories)
    fp = np.zeros(num_categories)
    fn = np.zeros(num_categories)
    out = torch.argmax(torch.softmax(pred, dim = 1), dim = 1).squeeze(0)

    prediction = out.cpu().detach().numpy()

    label = label.cpu()
    label = label.numpy()
    for cat in range(num_categories):
        tp[cat] += ((prediction == cat) & (label == cat) & (label < num_categories)).sum()
        fp[cat] += ((prediction == cat) & (label != cat) & (label < num_categories)).sum()
        fn[cat] += ((prediction != cat) & (label == cat) & (label < num_categories)).sum()

    np.seterr(divide = 'ignore', invalid = 'ignore')
    iou = np.divide(tp, tp + fp + fn)

    m = iou.mean()
    return m

def trainer_synapse(args, model, snapshot_path, print_per_iter = 100):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # uncomment to also print logs 
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    # Load training and validation data
    db_train = CustomDataLoader(args.root_path.format('train'), args.mask_path.format('train'),
                               height = args.img_size, width = args.img_size, data_type = 'float', expand_dims = False)
    db_val = CustomDataLoader(args.root_path.format('val'), args.mask_path.format('val'),
                               height = args.img_size, width = args.img_size, data_type = 'float', expand_dims = False)
    print(f"Length of train set:\t{len(db_train)}")
    print(f"Length of val set:\t{len(db_val)}")
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True,
                             worker_init_fn = worker_init_fn)
    val_loader = DataLoader(db_val, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True,
                             worker_init_fn = worker_init_fn)

    model.train()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) 

    writer = SummaryWriter(snapshot_path + '/log')
    
    iter_num = 0
    max_epoch = args.max_epochs

    max_iterations = args.max_epochs * len(train_loader)
    # max_epoch = max_iterations // len(train_loader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    best_performance = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # print('epoch:',epoch_num)
        epoch_loss = 0
        logging.info('epoch {}'.format(epoch_num))
        for sampled_batch in tqdm(train_loader):
            image_batch, label_batch = sampled_batch 
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            image_batch = image_batch.float()

            # Redundant reshape?
            # image_batch = image_batch.transpose([0, 3, 1, 2])

            outputs = model(image_batch) # Predict

            # Compute loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice  # loss
            
            # Sum the loss for all samples in the batch
            epoch_loss += loss * args.batch_size
    
            miou = cal(outputs, label_batch, args.num_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1 
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            #writer.add_scalar('info/miou', miou, iter_num)

            if iter_num % print_per_iter == 0: # Log the training performance for the current batch 
                logging.info('iteration %d : lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            
            # if iter_num % 1 == 0:
            #     image = image_batch[0, :, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
            #     labs = label_batch[0, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        val_loss = evaluate(model, val_loader, args)
        logging.info('train loss:\t{:.3f} (avg)\t {:.3f} (sum)'.format(epoch_loss/len(db_train), epoch_loss))
        logging.info('val loss:\t{:.3f} (avg)'.format(val_loss))
        
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num,
        }
        torch.save(checkpoint, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
