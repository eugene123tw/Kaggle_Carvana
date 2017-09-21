## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py
## https://github.com/pytorch/examples/blob/master/imagenet/main.py

from net.common import *
<<<<<<< HEAD
from net.model.loss import *
=======
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
from net.dataset.dataprocess import *
from net.dataset.mask import *
from net.rates import *
from net.util import *
<<<<<<< HEAD

# from net.model.unet import UNet512_2 as Net
# from net.model.unet import UNet512_shallow as Net
# from net.model.unet import UNet512_shallow_bilinear as Net
# from net.model.unet import UNet1024 as Net
# from net.model.networks import SegNet as Net
# from net.model.unet import UNet_pyramid_1 as Net
# from net.model.unet import UNet_pyramid_1024 as Net
from net.model.unet import UNet_pyramid_1024_2 as Net


import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from tensorboard_logger import configure, log_value

THRESHOLD = 127 # 0.5 # 127
RESUME = False
ADJUST_LR = False
SAVE_GBEST = True

=======
# from net.model.deeplab import VGG_deeplab as Net
# from net.model.vgg import vgg16 as Net
# from net.model.unet import UNet256 as Net
from net.model.unet import UNet512_2 as Net
# from net.model.unet import UNet512_shallow as Net
# from net.model.networks import SegNet as Net

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models
from torch.autograd import Variable


THRESHOLD = 128 #0.5
RESUME = True

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1)+m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score

def multiCriterion(logits, masks):
    l = BCELoss2d()(logits, masks) + SoftDiceLoss()(logits, masks)
    return l
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

############################################################################
def show_train_batch_results(probs, masks, images, indices, wait=1, save_dir=None, names=None):

    probs  = (probs.data.cpu().numpy().squeeze()*255).astype(np.uint8)
    masks = (masks.data.cpu().numpy()*255).astype(np.uint8)
    images = (images.data.cpu().numpy()*255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))

    batch_size,H,W,C = images.shape
    results = np.zeros((H, 3*W, 3),np.uint8)
    prob    = np.zeros((H, W, 3),np.uint8)
    for b in range(batch_size):
        m = probs [b]>128
        l = masks[b]>128
        score = one_dice_loss_py(m , l)

        image = images[b]
        prob[:,:,1] = probs [b]
        prob[:,:,2] = masks[b]

        results[:,  0:W  ] = image
        results[:,  W:2*W] = prob
        results[:,2*W:3*W] = cv2.addWeighted(image, 1, prob, 1., 0.) # image * α + mask * β + λ
        draw_shadow_text  (results, '%0.3f'%score, (5,15),  0.5, (255,255,255), 1)

        if save_dir is not None:
            shortname = names[indices[b]].split('/')[-1].replace('.jpg','')
            cv2.imwrite(save_dir + '/%s.jpg'%shortname, results)


        im_show('train',  results,  resize=1)
        cv2.waitKey(wait)


#https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
#https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score

#https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2 ):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = score.sum()/num
    return score


def predict(net, test_loader):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_H, CARVANA_W
    predictions  = np.zeros((num, H, W),np.float32)

    test_num  = 0
    for it, (images, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)

        batch_size = len(indices)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1, H, W)

    assert(test_num == len(test_loader.sampler))
    return predictions


<<<<<<< HEAD
=======


>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
def predict_asIntType(net, test_loader, out_dir):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_H, CARVANA_W
    # predictions = np.memmap(out_dir + '/predictions.npy', dtype=np.float32, mode='w+', shape=(num, H, W))
    predictions = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(num, H, W))

    test_num  = 0
    for it, (images, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)

        batch_size = len(indices)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        probs = probs.data.cpu().numpy().reshape(-1, H, W)*255
        predictions[start:end] = probs.astype(np.uint8)

    assert(test_num == len(test_loader.sampler))
    return predictions

<<<<<<< HEAD
def predict_asFloatType(net, test_loader, out_dir):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_H, CARVANA_W
    predictions = np.memmap(out_dir + '/preds.npy', dtype=np.float32, mode='w+', shape=(num, H, W))

    test_num  = 0
    for it, (images, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)

        batch_size = len(indices)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        probs = probs.data.cpu().numpy().reshape(-1, H, W)
        predictions[start:end] = probs.astype(np.float32)

    assert(test_num == len(test_loader.sampler))
    return predictions
=======
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

def evaluate(net, test_loader):

    sum_smooth_loss = 0.0
    test_acc = 0.0
    test_loss = 0.0
    sum=0

    for it, (images, masks, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())

        # forward
        logits = net(images)
        probs = F.sigmoid(logits)
        pred_masks = (probs > 0.5).float()
<<<<<<< HEAD
        # loss = criterion(logits, masks)
        loss = BCELoss2d()(logits, masks)

        # print statistics
=======
        loss = BCELoss2d()(logits, masks)

        # print statistics
        sum_smooth_loss += loss.data[0]
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
        test_acc += dice_loss(masks, pred_masks).data[0]
        test_loss += loss.data[0]
        sum += 1

<<<<<<< HEAD
    test_acc = test_acc / sum
    test_loss = test_loss / sum

    return test_acc, test_loss

=======
    smooth_loss = sum_smooth_loss / sum
    test_acc = test_acc / sum
    test_loss = test_loss / sum

    return test_acc,test_loss
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa


def ensemble():
    out_dir  = '/home/eugene/Documents/Kaggle_Carvana/results/ensemble'
    data_dir = '/home/eugene/Documents/Kaggle_Carvana/data'

    ensembleList = [
<<<<<<< HEAD
        '/home/eugene/Documents/Kaggle_Carvana/results/unet-1-640',
        '/home/eugene/Documents/Kaggle_Carvana/results/unet-1-1024',
        '/home/eugene/Documents/Kaggle_Carvana/results/unet-2-1024',
=======
        '/home/eugene/Documents/Kaggle_Carvana/results/unet-9-512',
        '/home/eugene/Documents/Kaggle_Carvana/results/unet-0-640',
        '/home/eugene/Documents/Kaggle_Carvana/results/unet-1-640',
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    ]

    imageSize = [item.split('-')[-1] for item in ensembleList]

    infoDict = dict(zip(range(len(ensembleList)),zip(imageSize, ensembleList)))

    infoDict = collections.OrderedDict(sorted(infoDict.items()))

    splitTxt = data_dir + '/split/' + 'test-INTER_LINEAR-640x640-100064'

    with open(splitTxt) as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    num_test = len(names)

    for n in range(num_test):
        name = names[n].split('/')[-1]
        names[n] = name.replace('<mask>', '').replace('<ext>', 'jpg')

    print('make csv')
    csv_file = out_dir + '/submit/ensemble.csv'
    gz_file = csv_file + '.gz'

    # verify file order is correct!
    start = timer()
    rles = []
    for n in range(num_test):
        if (n % 1000 == 0):
            end = timer()
            time = (end - start) / 60
            time_remain = (num_test - n - 1) * time / (n + 1)
<<<<<<< HEAD
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min' % (n, num_test, time, time_remain))
=======
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min' % (
                n, num_test, time, time_remain))
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

        for index, itemInfo in infoDict.items():
            item = np.memmap(itemInfo[1] + '/preds.npy', dtype=np.uint8, mode='r', shape=(100064, int(itemInfo[0]), int(itemInfo[0])))
            item = np.asarray(item[n], dtype=np.float32)
            if index==0:
                prob = cv2.resize(item, (CARVANA_WIDTH, CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                prob += cv2.resize(item, (CARVANA_WIDTH, CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR)

        prob = prob/len(ensembleList)
        mask = prob > THRESHOLD
        rle = run_length_encode(mask)

        prob = None
        mask = None
        rles.append(rle)

        if (n % 1000 == 0):
            print('Length of rles: %s' % (len(rles),))

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


def do_training():

    out_dir          = '/home/eugene/Documents/Kaggle_Carvana/results/debug-0'
    train_split      = 'debug-INTER_LINEAR-256-33' # 'train-INTER_LINEAR-640-4700' # 'train-INTER_LINEAR-640-5088' , 'train-INTER_LINEAR-512-5088' , 'train-INTER_LINEAR-512-4700'
    validation_split = 'debug-INTER_LINEAR-256-33'

    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/snap', exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')
    log.write('Training Split File = %s\n' % train_split)
    log.write('Validation Split File = %s\n' % validation_split)

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 2
    train_dataset = ImageDataset(train_split,
                                transform=[
                                    # lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(0,0)),
                                    # lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
                                is_mask=True,
                                is_preload=False,)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)

    test_dataset = ImageDataset( validation_split,
                                 # transform=[
                                 #     lambda x, y: randomShiftScaleRotate2(x, y, shift_limit=(-0.0625, 0.0625),scale_limit=(-0.1, 0.1), rotate_limit=(0, 0)),
                                 #     lambda x, y: randomHorizontalFlip2(x, y),
                                 # ],
                                  is_mask=True,
                                  is_preload=False,)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)

    H, W = CARVANA_H, CARVANA_W

    ## net ----------------------------------------
    log.write('** net setting **\n')

    # net = Net(in_shape=(3, H, W), num_classes=1)
    net = Net(num_classes=1)
    net.cuda().train()

    log.write('%s\n\n' % (type(net)))

    ## optimiser ----------------------------------
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005

    num_epoches = 35  # 100
    it_print = 1  # 20
    epoch_test = 1
    epoch_valid = 1
    epoch_save = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, num_epoches - 1]

    ## resume from previous ----------------------------------
    start_epoch = 0

    # training ####################################################################3
    log.write('** start training here! **\n')
    log.write('Trainig Batch Size= %s\n' % batch_size)
    log.write('\n')

    log.write('epoch    iter   rate | smooth_loss | train_loss train_acc | test_loss test_acc | \n')
    log.write('--------------------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss = np.nan
    train_acc = np.nan
    test_loss = np.nan
    test_acc = np.nan

    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        # print ('epoch=%d'%epoch)
        start = timer()

        # ---learning rate schduler ------------------------------
        if epoch >= 38:
            adjust_learning_rate(optimizer, lr=0.001)

        rate = get_learning_rate(optimizer)[0]  # check
        # --------------------------------------------------------


        sum_smooth_loss = 0.0
        sum = 0
        net.train()
        num_its = len(train_loader)
        for it, (images, masks, indices) in enumerate(train_loader, 0):

            images = Variable(images.cuda())
            masks = Variable(masks.cuda())

            # forward
            logits = net(images)
            # logits = F.upsample_bilinear(logits, (H, W))
            probs = F.sigmoid(logits)
            pred_masks = (probs > 0.5).float()

            # backward
            loss = BCELoss2d()(logits, masks)
            # loss = CrossEntropyLoss2d()(logits, masks)
            # loss = multiCriterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            sum_smooth_loss += loss.data[0]
            sum += 1
            if it % it_print == 0 or it == num_its - 1:
                smooth_loss = sum_smooth_loss / sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc = dice_loss(masks, pred_masks).data[0]
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | ... ' % \
                      (epoch + (it + 1) / num_its, it + 1, rate, smooth_loss, train_loss, train_acc),end='', flush=True)

                # debug show prediction results ---
            if 0:
                show_train_batch_results(probs, masks, images, indices,wait=1, save_dir=out_dir + '/train/results', names=train_dataset.names)

        end = timer()
        time = (end - start) / 60

        if epoch % epoch_valid == 0 or epoch == 0 or epoch == num_epoches - 1:
            net.cuda().eval()
            test_acc, test_loss = evaluate(net, test_loader)
            print('evalute\n',end='', flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' % \
                      (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss, test_acc, time))

        if epoch in epoch_save:
            torch.save(net.state_dict(),out_dir +'/snap/%03d.pth'%epoch)
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/checkpoint/%03d.pth'%epoch)
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py

    # ---- end of all epoches -----
    end0 = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-test
    torch.save(net.state_dict(), out_dir + '/snap/final.pth')


def do_training_accu_gradients():

<<<<<<< HEAD
    out_dir          = '/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-4-1024'
    train_split      = 'train-fold-4'
    validation_split = 'val-fold-4'

    # Create dirs ---------------------------------------------------------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/snap', exist_ok=True)
    os.makedirs(out_dir + '/tb_logging', exist_ok=True)
    # ---------------------------------------------------------------------------
=======
    out_dir          = '/home/eugene/Documents/Kaggle_Carvana/results/unet-1-1024'
    train_split      = 'train-INTER_LINEAR-1024-4700' # 'train-INTER_LINEAR-640-4700' # 'train-INTER_LINEAR-640-5088' , 'train-INTER_LINEAR-512-5088' , 'train-INTER_LINEAR-512-4700'
    validation_split = 'train-INTER_LINEAR-1024-388'
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/snap', exist_ok=True)
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

    batch_size = 2     # Define batch size
    accmulate_size = 4 # Define accumulate size

<<<<<<< HEAD

    # Initial tensorboard logger ---------------------------------------------------------------
    configure(out_dir + '/tb_logging', flush_secs=5)
    # --------------------------------------------------------------------------------------------

=======
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    # Initial log ------------------------------------------------------------------------------
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')
    log.write('Training Split File = %s\n' % train_split)
    log.write('Validation Split File = %s\n' % validation_split)
    log.write('** dataset setting **\n')
    # --------------------------------------------------------------------------------------------

    # Prepare DataLoader -----------------------------------------------------
    train_dataset = ImageDataset(train_split,
                                transform=[
<<<<<<< HEAD
                                    lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(0,0), scale_limit=(-0.5,0.5), rotate_limit=(0,0)),
                                    # lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
                                type = 'train',
                                resize= False,
                                folder= 'train-INTER_LINEAR-1024x1024-hq',
                                mask_folder= 'mask-INTER_LINEAR-1024x1024',
                                )
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSamplerWithLength(train_dataset, 4580),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
=======
                                    # lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(0,0)),
                                    # lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
                                is_mask=True,
                                is_preload=False,)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 0,
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
                        pin_memory  = True)

    test_dataset = ImageDataset( validation_split,
                                 # transform=[
                                 #     lambda x, y: randomShiftScaleRotate2(x, y, shift_limit=(-0.0625, 0.0625),scale_limit=(-0.1, 0.1), rotate_limit=(0, 0)),
                                 #     lambda x, y: randomHorizontalFlip2(x, y),
                                 # ],
<<<<<<< HEAD
                                 type='train',
                                 resize=False,
                                 folder='train-INTER_LINEAR-1024x1024-hq',
                                 mask_folder='mask-INTER_LINEAR-1024x1024',
                                 )
=======
                                  is_mask=True,
                                  is_preload=False,)
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
<<<<<<< HEAD
                        num_workers = 6,
=======
                        num_workers = 0,
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
                        pin_memory  = True)
    # -----------------------------------------------------------------------------

    H, W = CARVANA_H, CARVANA_W

    # Set Net ---------------------------------
    ## resume from previous ----------------------------------
    if RESUME is True :
<<<<<<< HEAD
        model_file = '/home/eugene/Documents/Kaggle_Carvana/results/unet-py-1/checkpoint/gbest.pth'
        # net = Net(in_shape=(3, H, W), num_classes=1)  # for U-Net model
        net = Net(in_shape=(3, H, W))
        # net = Net(num_classes=1)
        checkpoint = torch.load(model_file)
        start_epoch = checkpoint['epoch']
        # start_epoch = 0
        net.load_state_dict(checkpoint['state_dict'])

        ## optimiser ---------------------------------------------------------
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
        optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0)
        optimizer.load_state_dict(checkpoint['optimizer'])

        if ADJUST_LR:
            adjust_learning_rate(optimizer, lr=0.01)

        net.cuda().train()

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    else:
        # Train from scratch
        start_epoch = 0
        net = Net(in_shape=(3, H, W)) # for U-Net model
        # net = Net(in_shape=(3, H, W))
        # net = Net(num_classes=1)
=======

        model_file = out_dir +'/checkpoint/010.pth'  #final
        net = Net(in_shape=(3, H, W), num_classes=1)

        checkpoint = torch.load(model_file)

        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])

        ## optimiser ---------------------------------------------------------
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
        # optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.cuda().train()
    else:
        # Train from scratch
        start_epoch = 0
        net = Net(in_shape=(3, H, W), num_classes=1) # for U-Net model
        # net = Net(num_classes=1)                     # For other model
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
        net.cuda().train()

        ## optimiser ---------------------------------------------------------
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
<<<<<<< HEAD
        optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

=======
        optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    # -----------------------------------------


    # --------------------------------------------------------------------

    ## Epoch and Iterator Variables ------------------------------------------------
<<<<<<< HEAD
    num_epoches = 50  # 100
    it_print = 1  # 20
    epoch_test = 1
    epoch_valid = 1
    epoch_save = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, num_epoches - 1]
    ## --------------------------------------------------------------------


    # Prepare training log ------------------------------------------------------------------------------
    log.write('** start training here! **\n')
    log.write('num_grad_acc x batch_size = %d x %d = %d\n' % (accmulate_size, batch_size, accmulate_size * batch_size))
    log.write('Trainig Batch Size= %s\n' % batch_size)
    log.write('\n')

    log.write(' epoch    iter   rate   | valid_loss  valid_acc | train_loss  train_acc | batch_loss  batch_acc \n')
    log.write('-----------------------------------------------------------------------------------------------\n')
    # --------------------- -----------------------------------------------------------------------------

    # Initialize variables -------------
    accu_counter = 0
    batch_loss_accumul = 0.0
    batch_acc_accumul = 0.0
    batch_loss = 0.0
    batch_acc = 0.0
    train_loss = np.nan
    train_acc = np.nan
    gbest = 0 # Global best
    test_loss = 0
=======
    num_epoches = 35  # 100
    it_print = 1  # 20
    epoch_test = 1
    epoch_valid = 1
    epoch_save = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, num_epoches - 1]
    ## --------------------------------------------------------------------




    # Prepare training log ------------------------------------------------------------------------------
    log.write('** start training here! **\n')
    log.write('Trainig Batch Size= %s\n' % batch_size)
    log.write('\n')

    log.write('epoch    iter   rate | smooth_loss | train_loss train_acc | test_loss test_acc | \n')
    log.write('--------------------------------------------------------------------------------------------------\n')
    # --------------------- -----------------------------------------------------------------------------

    # Initialize variables -------------
    sum = 0                 # for smooth loss counter, increase during each epoch
    sum_smooth_loss = 0.0   # summarize loss for each epoch
    smooth_loss = 0.0       # for printing smooth_loss
    accu_counter = 0
    loss_accumul = 0.0
    train_accuracy_accumul = 0.0
    train_loss = np.nan
    train_acc = np.nan
    test_loss = np.nan
    test_acc = np.nan
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    # -----------------------------------
    start0 = timer()

    for epoch in range(start_epoch, num_epoches):

        start = timer() # Start timer

        # ---learning rate schduler ------------------------------
<<<<<<< HEAD
        # if epoch > 24:
        #     adjust_learning_rate(optimizer, lr=0.0001)
        # elif epoch > 9:
        #     adjust_learning_rate(optimizer, lr=0.001)
        #
        # rate = get_learning_rate(optimizer)[0]  # check

        if epoch > 0:
            scheduler.step(test_loss)
        rate = get_learning_rate(optimizer)[0]
=======
        if epoch >= 10:
            adjust_learning_rate(optimizer, lr=0.001)
        elif epoch >= 25:
            adjust_learning_rate(optimizer, lr=0.0001)

        rate = get_learning_rate(optimizer)[0]  # check
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
        # --------------------------------------------------------

        net.train() # Set model.train() after model.eval()
        num_its = len(train_loader)

        for it, (images, masks, indices) in enumerate(train_loader, 0):
<<<<<<< HEAD
=======

>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
            # Set accumulator
            accu_counter += 1

            # Insert Variables in GPU -----------------------------
            images = Variable(images.cuda())
            masks  = Variable(masks.cuda())
            # -----------------------------------------------------

            # Forward pass and compute loss -----------------------
            logits = net(images)
<<<<<<< HEAD
            loss = criterion(logits, masks)
            # loss = BCELoss2d()(logits, masks)
=======
            loss = BCELoss2d()(logits, masks)
            loss_accumul += loss.data[0]
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
            loss.backward() # Accumulate gradients
            # -----------------------------------------------------

            # Compute predict masks -------------------------------
            probs = F.sigmoid(logits)
            pred_masks = (probs > 0.5).float()
            # -----------------------------------------------------

<<<<<<< HEAD
            # Train loss and Train acc ----------------------------
            train_loss = loss.data[0]
            train_acc  = dice_loss(masks, pred_masks).data[0]

            batch_loss_accumul += train_loss
            batch_acc_accumul  += train_acc
            # -----------------------------------------------------
=======
            train_accuracy_accumul += dice_loss(masks, pred_masks).data[0]

>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

            if accu_counter == accmulate_size:
                # (1.) Update weights And (2.) Computer train loss
                optimizer.step() # Update weights
<<<<<<< HEAD
                batch_loss = batch_loss_accumul / accmulate_size  # Average loss
                batch_acc = batch_acc_accumul / accmulate_size

                # Re-initialize -------------------------------------
                optimizer.zero_grad() # zero gradients
                batch_loss_accumul = 0
                batch_acc_accumul  = 0
                accu_counter       = 0

                print("\r%5.1f   %5d    %0.4f   | ........  .........    | %0.5f    %0.5f | %0.5f    %0.5f | " % \
                      (epoch + (it + 1) / num_its, it + 1, rate, train_loss, train_acc, batch_loss, batch_acc), end="", flush=False)
=======
                train_loss = loss_accumul / accmulate_size  # Average loss
                train_acc  = train_accuracy_accumul / accmulate_size
                # Compute smooth loss -------------------------------------
                sum_smooth_loss += train_loss
                sum += 1
                smooth_loss = sum_smooth_loss / sum
                # ---------------------------------------------------------

                print('\r %5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | ... ' % \
                      (epoch + (it + 1) / num_its, it + 1, rate, smooth_loss, train_loss, train_acc), end='',flush=True)

                # Re-initialize -------------------------------------
                optimizer.zero_grad() # zero gradients

                loss_accumul           = 0
                train_accuracy_accumul = 0
                accu_counter           = 0
                # ----------------------------------------------------
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa


        # Re-initialize -------------------------------------
        optimizer.zero_grad()  # zero gradients
<<<<<<< HEAD
=======
        loss_accumul = 0
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
        accu_counter = 0
        # ----------------------------------------------------

        end = timer()
        time = (end - start) / 60

        if epoch % epoch_valid == 0 or epoch == 0 or epoch == num_epoches - 1:
<<<<<<< HEAD
            net.eval()
            test_acc, test_loss = evaluate(net, test_loader)
            print('\r', end='', flush=True)
            log.write('%5.1f   %5d    %0.4f   | %0.5f      %0.5f   | %0.5f    %0.5f | %0.5f    %6.5f | %3.1f min \n' % \
                      (epoch + (it + 1) / num_its, it + 1, rate, test_loss, test_acc, train_loss, train_acc, batch_loss, batch_acc,  time))

            log_value('Learning Rate', rate, epoch)
            log_value('Test Loss', test_loss, epoch)
            log_value('Test Accurary', test_acc, epoch)

        if SAVE_GBEST:
            if test_acc > gbest:
                gbest = test_acc
                torch.save(net.state_dict(), out_dir + '/snap/gbest.pth')
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, out_dir + '/checkpoint/gbest.pth')
=======
            net.cuda().eval()
            test_acc, test_loss = evaluate(net, test_loader)
            log.write('\n %5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' % \
                      (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss, test_acc, time))
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

        if epoch in epoch_save:
            torch.save(net.state_dict(), out_dir + '/snap/%03d.pth' % epoch)
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, out_dir + '/checkpoint/%03d.pth' % epoch)

    # ---- end of all epoches -----
    end0 = timer()
    time0 = (end0 - start0) / 60
    log.write('Total time: %d' %(time0,))

    ## check : load model and re-test
    torch.save(net.state_dict(), out_dir + '/snap/final.pth')



##-----------------------------------------
def do_submissions():

    out_dir = '/home/eugene/Documents/Kaggle_Carvana/results/unet-4-512'
    model_file = out_dir +'/snap/final.pth'  #final

    # logging, etc --------------------
    os.makedirs(out_dir + '/submit/results', exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 8

    test_dataset = ImageDataset( 'test-INTER_LINEAR-512X512-50000', #'test-INTER_LINEAR-512x512-50024', #'test-INTER_LINEAR-256x256-50000', # 'test-INTER_LINEAR-256x256-50024', #'test-256x256-50024', # 'test-256x256-50000',
                                is_mask=False,
                                is_preload=False,  # True,
                                type='test',
                                )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
        pin_memory=True)

    H, W = CARVANA_H, CARVANA_W

    ## net ----------------------------------------
    net = Net(in_shape=(3, H, W), num_classes=1)
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    ## start testing now #####
    log.write('start prediction ...\n')
    if 1:
        net.eval()
        # probs = predict(net, test_loader)
        probs = predict_asIntType(net, test_loader)
        np.save(out_dir + '/submit/probs.npy', probs)
    else:
        probs = np.load(out_dir + '/submit/probs.npy')

    if 0:
        num = 500
        results = np.zeros((H, 3 * W, 3), np.uint8)
        prob = np.zeros((H, W, 3), np.uint8)
        num_test = len(probs)
        for n in range(num):
            shortname = test_dataset.names[n].split('/')[-1].replace('.jpg', '')
            image, index = test_dataset[n]
            image = tensor_to_image(image, std=255)
            prob[:, :, 1] = probs[n] * 255

            results[:, 0:W] = image
            results[:, W:2 * W] = prob
            results[:, 2 * W:3 * W] = cv2.addWeighted(image, 0.75, prob, 1., 0.)  # image * α + mask * β + λ

            cv2.imwrite(out_dir + '/submit/results/%s.jpg' % shortname, results)
            im_show('test', results, resize=1)
            cv2.waitKey(1)

    # resize to original and make csv

    print('make csv')
    csv_file = out_dir + '/submit/results-debug.csv'
    gz_file = csv_file + '.gz'

    # verify file order is correct!
    num_test = len(test_dataset.names)
    names = []

    for n in range(num_test):
        name = test_dataset.names[n].split('/')[-1]
        name = name.replace('<mask>','').replace('<ext>','jpg')
        names.append(name)

    start = timer()
    rles = []
    for n in range(num_test):
        if (n % 1000 == 0):
            end = timer()
            time = (end - start) / 60
            time_remain = (num_test - n - 1) * time / (n + 1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min' % (
            n, num_test, time, time_remain))

        prob = probs[n]
        prob = cv2.resize(prob, (CARVANA_WIDTH, CARVANA_HEIGHT))
        mask = prob > THRESHOLD
        rle = run_length_encode(mask)
        rles.append(rle)

        # im_show('prob', prob*255, resize=0.333)
        # cv2.waitKey(0)

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


def do_submit_efficient_mapping():
<<<<<<< HEAD
    out_dir = '/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-2-1024'
    model_file = out_dir + '/snap/final.pth'  # final
    # model_file = out_dir + '/snap/gbest.pth'  # final
    # model_file = out_dir + '/snap/010.pth'  # final

=======
    out_dir = '/home/eugene/Documents/Kaggle_Carvana/results/unet-1-1024'
    model_file = out_dir + '/snap/final.pth'  # final
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

    # logging, etc --------------------
    os.makedirs(out_dir + '/submit/results', exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
<<<<<<< HEAD
    batch_size = 6

    ## Used model
    log.write("Model: %s \n" % (model_file))

    test_dataset = ImageDataset('test-hq-100064',
                                type='test',
                                folder = 'test-INTER_LINEAR-1024x1024-hq',
=======
    batch_size = 4

    test_dataset = ImageDataset('test-INTER_LINEAR-1024x1024-100064',
                                # 'test-INTER_LINEAR-512x512-50024', #'test-INTER_LINEAR-256x256-50000', # 'test-INTER_LINEAR-256x256-50024', #'test-256x256-50024', # 'test-256x256-50000',
                                is_mask=False,
                                is_preload=False,  # True,
                                type='test',
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
                                )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
        pin_memory=True)

    H, W = CARVANA_H, CARVANA_W

    ## net ----------------------------------------
    # net = Net(in_shape=(3, H, W), num_classes=1)
<<<<<<< HEAD
    # net = Net(num_classes=1)
    net = Net(in_shape=(3, H, W))
=======
    net = Net(num_classes=1)
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    ## start testing now #####
    log.write('start prediction ...\n')

    if 1:
        net.eval()
        # probs = predict_efficient(net, test_loader, out_dir)
        probs = predict_asIntType(net, test_loader, out_dir)
<<<<<<< HEAD
        # probs = predict_asFloatType(net, test_loader, out_dir)
=======
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
        # np.save(out_dir + '/submit/probs-2.npy', probs)


        # resize to original and make csv

    print('make csv')
<<<<<<< HEAD
    csv_file = out_dir + '/submit/results-final.csv'
=======
    csv_file = out_dir + '/submit/results-th%03d.csv' % THRESHOLD
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
    gz_file = csv_file + '.gz'

    # verify file order is correct!
    num_test = len(test_dataset.names)
    names = []

    for n in range(num_test):
        name = test_dataset.names[n].split('/')[-1]
<<<<<<< HEAD
        name = name.replace('<mask>', '').replace('<ext>', 'jpg').replace('png', 'jpg')
=======
        name = name.replace('<mask>', '').replace('<ext>', 'jpg')
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
        names.append(name)

    start = timer()
    rles = []
    for n in range(num_test):
        if (n % 1000 == 0):
            end = timer()
            time = (end - start) / 60
            time_remain = (num_test - n - 1) * time / (n + 1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min' % (
                n, num_test, time, time_remain))

        prob = probs[n]
        prob = cv2.resize(prob, (CARVANA_WIDTH, CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR)
        mask = prob > THRESHOLD
        rle = run_length_encode(mask)
        rles.append(rle)

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


<<<<<<< HEAD
#decode and check
def run_check_submit_csv():

    gz_file = '/home/eugene/Documents/Kaggle_Carvana/results/unet-3-1024/submit/results-th128.csv.gz'
    df = pd.read_csv(gz_file, compression='gzip', header=0)

    # indices =[0,1,2,32000-1,32000,32000+1,100064-1]
    indices = np.random.choice(100064, 30, replace=False)

    for n in indices:
        name = df.values[n][0]
        img_file = KAGGLE_DATA_DIR +'/image/test-jpg/%s'%name
        img = cv2.imread(img_file)
        im_show('img', img, resize=0.25)
        plt.waitforbuttonpress()

        rle   = df.values[n][1]
        mask  = run_length_decode(rle,H=CARVANA_HEIGHT, W=CARVANA_WIDTH)
        im_show('mask', mask, resize=0.25, cmap='gray')
        plt.waitforbuttonpress()

    pass




def prediction_to_csv():
    out_dir     = '/home/eugene/Documents/Kaggle_Carvana/results/unet-2-1024'
    predict_dir = '/home/eugene/Documents/Kaggle_Carvana/results/unet-2-1024/pred_mask_crf'

    ## dataset ----------------------------

    test_dataset = ImageDataset('test-INTER_LINEAR-1024x1024-100064',
                                # 'test-INTER_LINEAR-512x512-50024', #'test-INTER_LINEAR-256x256-50000', # 'test-INTER_LINEAR-256x256-50024', #'test-256x256-50024', # 'test-256x256-50000',
                                is_mask=False,
                                is_preload=False,  # True,
                                type='test',
                                )

    print('make csv')
    csv_file = out_dir + '/submit/results-th%03d.csv' % THRESHOLD
    gz_file = csv_file + '.gz'

    # verify file order is correct!
    num_test = len(test_dataset.names)
    names = []

    for n in range(num_test):
        name = test_dataset.names[n].split('/')[-1]
        name = name.replace('<mask>', '').replace('<ext>', 'jpg')
        names.append(name)

    start = timer()
    rles = []
    for n in range(num_test):
        if (n % 1000 == 0):
            end = timer()
            time = (end - start) / 60
            time_remain = (num_test - n - 1) * time / (n + 1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min' % (
                n, num_test, time, time_remain))

        prob = cv2.imread(predict_dir + '/' + names[n],0)
        mask = prob > THRESHOLD
        rle = run_length_encode(mask)
        rles.append(rle)

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')



# main #################################################################
if __name__ == '__main__':


    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_training_accu_gradients()
    # do_submit_efficient_mapping()

    # ensemble()
    # run_check_submit_csv()
    # run_csv_to_mask()

    # run_csv_to_mask()

    # prediction_to_csv()
=======
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # do_training()
    # do_submissions()

    # do_training_accu_gradients()
    do_submit_efficient_mapping()


    # ensemble()

    if 0:
        net = Net(in_shape=(3, 128, 128), num_classes=1)
        net.cuda()

        pretrained_file = '/home/eugene/Documents/Kaggle_Carvana/data/pretrain/vgg16-397923af.pth'
        skip_list = ['fc.0.weight', 'fc.0.bias', 'fc.3.weight', 'fc.3.bias', 'fc.6.weight', 'fc.6.bias']
        # skip_list = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
        pretrained_dict = torch.load(pretrained_file)
        model_dict = net.state_dict()

        pretrained_dict1 = {k:v for k,v in pretrained_dict.items() if k in model_dict and k not in skip_list}
        # pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict1)

        net.load_state_dict(model_dict)

        print('\nhold!')
        pass

>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

    print('\nsucess!')