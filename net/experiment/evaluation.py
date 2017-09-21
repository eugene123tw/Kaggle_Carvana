
from net.common import *
from net.model.loss import *
from net.dataset.dataprocess import *
from net.dataset.mask import *
from net.rates import *
from net.util import *

from net.model.unet import UNet512_shallow as Net
# from net.model.unet import UNet512_shallow_bilinear as Net

def evaluate_real_acc(net, test_loader):

    test_acc = 0.0
    accumulator = 0
    sum=0

    for it, (images, masks, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())

        # forward
        logits = net(images)
        probs = F.sigmoid(logits)
        probs = F.upsample(probs, size=(CARVANA_HEIGHT, CARVANA_WIDTH), mode='bilinear')
        pred_masks = (probs > 0.5).float()

        # print statistics
        test_acc += dice_loss(masks, pred_masks).data[0]
        sum += 1

        accumulator += len(indices)

        if sum % 100 ==0:
            print("Progress: %06d/%06d" % (accumulator, len(test_loader)))



    test_acc = test_acc / sum

    return test_acc


if __name__ == '__main__':

    out_dir = '/home/eugene/Documents/Kaggle_Carvana/results/unet-7-1024'
    model_file = out_dir + '/snap/gbest-9962.pth'  # final

    validation_split = 'train-hq-5088'

    H, W = CARVANA_H, CARVANA_W

    test_dataset = ImageDataset(validation_split,
                                # transform=[
                                #     lambda x, y: randomShiftScaleRotate2(x, y, shift_limit=(-0.0625, 0.0625),scale_limit=(-0.1, 0.1), rotate_limit=(0, 0)),
                                #     lambda x, y: randomHorizontalFlip2(x, y),
                                # ],
                                type='train',
                                resize=False,
                                folder='train-INTER_LINEAR-1024x1024',
                                mask_folder='mask-jpg',
                                )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=2,
        pin_memory=True)

    net = Net(in_shape=(3, H, W), num_classes=1)
    net.load_state_dict(torch.load(model_file))
    net.cuda().eval()

    eval_acc = evaluate_real_acc(net, test_loader)
    print("Evaluation accuracy: %0.5f" % (eval_acc))


    pass