from net.common import *
from PIL import Image
import cv2
from multiprocessing import Pool


# INPUT_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image/mask-png'
INPUT_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image/test_hq'
# INPUT_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image/seg_images_2/images'
# INPUT_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image/seg_images_2/labels'
# INPUT_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image/train_hq'
ROOT_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image'

# draw -----------------------------------
def im_show(name, image, resize=1, cmap = ''):
    H,W = image.shape[0:2]
    if cmap=='gray':
        plt.imshow(image.astype(np.uint8),cmap='gray')
    plt.imshow(image.astype(np.uint8))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)


def draw_mask(image, mask, color=(255,255,255), α=1,  β=0.25, λ=0., threshold=32 ):
    # image * α + mask * β + λ

    if threshold is None:
        mask = mask/255
    else:
        mask = clean_mask(mask,threshold,1)

    mask  = np.dstack((color[0]*mask,color[1]*mask,color[2]*mask)).astype(np.uint8)
    image[...] = cv2.addWeighted(image, α, mask, β, λ)



## custom data transform  -----------------------------------

def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image*std + mean
    image = image.astype(dtype=np.uint8)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return image


def tensor_to_mask(tensor):
    label = tensor.numpy()*255
    label = label.astype(dtype=np.uint8)
    return label

## transform (input is numpy array, read in by cv2)
def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image-mean)/std
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)   ##.float()
    return tensor

def mask_to_tensor(mask, threshold=0.5):
    mask  = mask
    mask  = (mask>threshold).astype(np.float32)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    # tensor = torch.from_numpy(mask).type(torch.LongTensor)
    return tensor


def randomHorizontalFlip2(image, mask, u=0.5):

    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
        mask = cv2.flip(mask,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right

    return image, mask


def randomShiftScaleRotate2(image, mask, shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(-45,45), borderMode=cv2.BORDER_CONSTANT, u=0.5):
    #cv2.BORDER_REFLECT_101

    if random.random() < u:
        height,width,channel = image.shape

        angle = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale = random.uniform(1+scale_limit[0],1+scale_limit[1])
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(scale)
        ss = math.sin(angle/180*math.pi)*(scale)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return image,mask


#return fix data for debug #####################################################3
class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples

# see trorch/utils/data/sampler.py
class RandomSamplerWithLength(Sampler):
    def __init__(self, data, length):
        self.num_samples = length
        self.len_data= len(data)

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        l = list(range(self.len_data))
        random.shuffle(l)
        l= l[0:self.num_samples]
        return iter(l)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples


def imresize(inputName, destinationFolder='test-INTER_LINEAR-1024x1024-hq', width=1024, height=1024):

    input_img = INPUT_DIR+'/'+inputName
    outputName, extension = inputName.split('.') # xxxx.gif

    if extension == 'gif':
        out_dir = os.path.join(ROOT_DIR, destinationFolder, outputName + '.' + 'jpg')  # ROOT_DIR/folder/xxxx.jpg
        img = Image.open(input_img)
        img = np.array(img,dtype=np.float32)
    else:

        out_dir = os.path.join(ROOT_DIR, destinationFolder, outputName+'.'+'png') # ROOT_DIR/folder/xxxx.jpg
        img = cv2.imread(input_img, 1)

    print(out_dir)
    # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_dir, img)

def rename_file(dir, old_name, new_name):
    os.rename(old_name, new_name)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    if 1:
        filelist = os.listdir(INPUT_DIR)
        p = Pool(7)
        print(p.map(imresize, filelist))

        # for file in filelist:
        #   imresize(file)

    if 0:
        img_dir = '/home/eugene/Documents/Kaggle_Carvana/data/image/train-jpg'
        mask_dir = '/home/eugene/Documents/Kaggle_Carvana/data/image/mask-jpg'
        filelist = os.listdir(mask_dir)

        for item in filelist:
            input_img_dir = img_dir + '/' + item.replace('_mask','')
            input_mask_dir = mask_dir + '/' + item
            input_img = cv2.imread(input_img_dir)
            input_mask = cv2.imread(input_mask_dir)
            out_img, out_mask = randomShiftScaleRotate2(input_img, input_mask, shift_limit=(0,0), scale_limit=(-0.5,0.5), rotate_limit=(0,0))

            f, axarr = plt.subplots(2, sharex=True)

            axarr[0].imshow(out_img)
            axarr[1].imshow(out_mask)
            plt.waitforbuttonpress()
            # plt.imshow(out_img)
            # plt.waitforbuttonpress()
            # plt.imshow(out_mask)
            # plt.waitforbuttonpress()