from net.common import *
from net.dataset.tool import *
import pandas as pd
from PIL import Image

KAGGLE_DATA_DIR ='/home/eugene/Documents/Kaggle_Carvana/data'
DEBUG = False
CARVANA_HEIGHT = 1280
CARVANA_WIDTH  = 1918
<<<<<<< HEAD
CARVANA_H = 1024 # 640 # 1024 #640 # 684  # 256 # 1024 # 256
CARVANA_W = 1024 # 640 # 1024 #640 # 1024 # 256 # 1024 # 256
# 684 x 1024

class ImageDataset(Dataset):

    def __init__(self, split, folder, mask_folder=None , transform=[], type='train', resize = False):
        super(ImageDataset, self).__init__()

        # read names
        split_file = KAGGLE_DATA_DIR +'/split/'+ split
        with open(split_file) as f:
            names = f.readlines()
        names = [name.strip()for name in names]
        num   = len(names)

        # meta data
        # df = pd.read_csv(CARVANA_DIR +'/metadata.csv')

        #save
        # self.df        = df
        self.split     = split
        self.transform = transform

        self.names       = names
        self.type        = type
        self.resize      = resize
        self.folder      = folder
        self.mask_folder = mask_folder


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def get_image(self,index):
        folder = self.folder
        name   = self.names[index]

        if self.resize:
            name = name.replace('<ext>', 'jpg')
            img_file = KAGGLE_DATA_DIR + '/image/%s/%s' % (folder,name)
            img = cv2.imread(img_file, 1)
            img = cv2.resize(img, (CARVANA_W, CARVANA_H), interpolation=cv2.INTER_LINEAR)
        else :
            name = name.replace('<type>', self.type).replace('<mask>', '').replace('<ext>', 'png')
            img_file = KAGGLE_DATA_DIR + '/image/%s/%s' % (folder, name)
            img = cv2.imread(img_file)

        image = img.astype(np.float32)/255
        return image

    def get_label(self,index):
        name   = self.names[index]
        mask_folder = self.mask_folder

        if self.resize:
            name = name.replace('.', '_mask.').replace('<ext>', 'png')
            mask_file = KAGGLE_DATA_DIR + '/image/mask-png/%s' % (name)
            mask = cv2.imread(mask_file, 1)
            mask = cv2.resize(mask, (CARVANA_W, CARVANA_H), interpolation=cv2.INTER_LINEAR)
            mask = mask[:,:,0]
        else :
            name = name.replace('.', '_mask.').replace('<ext>', 'png')
            mask_file = KAGGLE_DATA_DIR + '/image/%s/%s' % (mask_folder, name)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        label = mask.astype(np.float32)/255
        return label

    def get_train_item(self,index):
        image = self.get_image(index)
        label = self.get_label(index)

        for t in self.transform:
            image,label = t(image,label)
        image = image_to_tensor(image)
        label = mask_to_tensor(label)
        return image, label, index

    def get_test_item(self,index):
        image = self.get_image(index)

        for t in self.transform:
            image = t(image)
        image = image_to_tensor(image)
        return image, index


    def __getitem__(self, index):

        if self.type=='train': return self.get_train_item(index)
        if self.type=='test':  return self.get_test_item (index)
=======
CARVANA_H = 1024 # 1024 # 256 # 640
CARVANA_W = 1024 # 1024 # 256 # 640


class ImageDataset(Dataset):

    def __init__(self, split, transform=[], height=CARVANA_H, width=CARVANA_W, is_mask=True, is_preload=True, type='train'):
        data_dir    = KAGGLE_DATA_DIR
        channel = 3
        # read names
        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip()for x in names]
        num   = len(names)

        images  = None
        if is_preload==True:
            images = np.zeros((num,height,width,channel),dtype=np.float32)
            for n in range(num):
                name = names[n].replace('<type>',type).replace('<mask>','').replace('<ext>','jpg')
                images[n] = cv2.imread(KAGGLE_DATA_DIR+'/image/'+name)/255.
                # images[n] = cv2.imread(KAGGLE_DATA_DIR + '/image/' + name)


        masks = None
        if is_mask ==True:
            masks = np.zeros((num, height, width),dtype=np.float32)
            # masks = np.zeros((num, 64, 64), dtype=np.float32)
            for n in range(num):
                name = names[n].replace('<type>', 'mask').replace('<mask>', '_mask').replace('<ext>', 'jpg')
                masks[n] = cv2.imread(KAGGLE_DATA_DIR + '/image/' + name, cv2.IMREAD_GRAYSCALE)/255.

        #save
        self.transform = transform
        self.names  = names
        self.images = images
        self.masks  = masks
        self.type = type


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        if self.images is None:
            name = self.names[index].replace('<type>',self.type).replace('<mask>','').replace('<ext>','jpg')
            img_file = KAGGLE_DATA_DIR+'/image/'+name
            img   = cv2.imread(img_file)
            image = img.astype(np.float32)/255.
            # image = img.astype(np.float32)
        else:
            image = self.images[index]


        if self.masks is None:
            for t in self.transform:
                image = t(image)
            image = image_to_tensor(image)
            return image, index
        else:
            mask = self.masks[index]
            for t in self.transform:
                image,mask = t(image, mask)
            image = image_to_tensor(image)
            mask = mask_to_tensor(mask)
            return image, mask, index
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa

    def __len__(self):
        return len(self.names)


def check_dataset(dataset, loader):

    if dataset.masks is not None:
        for i, (images, masks, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            num = len(images)
            for n in range(num):
                image = images[n]
                mask = masks[n]
                image = tensor_to_image(image, std=255)
                mask = tensor_to_mask(mask)

                im_show('image', image, resize=1)
                im_show('mask', mask, resize=1)



#test dataset
def run_check_dataset():
<<<<<<< HEAD
    dataset = ImageDataset( 'debug-INTER_LINEAR-256-500',  #'train_5088'
=======
    dataset = ImageDataset( 'debug-256-val',  #'train_5088'
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
                                transform=[
                                    lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(0,0)),
                                    lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
<<<<<<< HEAD
                            type = 'train',
=======
                            is_preload=False,
>>>>>>> 272eded3805ca69c6d80c862772ff4154780eefa
                         )

    if 1: #check indexing
        for n in range(100):
            image, mask, index = dataset[n]
            image = tensor_to_image(image, std=255)
            mask = tensor_to_mask(mask)

            im_show('image', image, resize=1)
            im_show('mask', mask, resize=1)

    if 0: #check iterator
        #sampler = FixedSampler(dataset, ([4]*100))
        sampler = SequentialSampler(dataset)
        loader  = DataLoader(dataset, batch_size=4, sampler=sampler,  drop_last=False, pin_memory=True)

        for epoch in range(100):
            print('epoch=%d -------------------------'%(epoch))
            check_dataset(dataset, loader)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()

    print('\nsucess!')