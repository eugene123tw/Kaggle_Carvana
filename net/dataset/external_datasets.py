from net.common import *
import matplotlib.pyplot as plt


EXTERNAL_DATA = "/Users/Eugene/Downloads"
IMAGE_FOLDER  = "/images/"
LABEL_FOLDER  = "/labels/"

OUT_FOLDER = "/Users/Eugene/Downloads/seg_images_2"

IMAGE_HEIGHT = 1052
IMAGE_WIDTH  = 1914

DEBUG = True


OBJ_DICT = {
'car':   [ 142, 0, 0],
'truck': [ 70,  0, 0],
'bus':   [ 100,  60, 0],
}



class ExternalData:

    def __init__(self, load_images = False, load_masks = False, height = IMAGE_HEIGHT , width = IMAGE_WIDTH):

        self.classes = [
            'Car', 'SUVPickupTruck', 'Truck_Bus',
        ]

        self.colors = [
            [64, 0, 128],
            [64, 128, 192],
            [192, 128, 192],
        ]

        self.names = os.listdir(EXTERNAL_DATA+IMAGE_FOLDER)
        self.len   = len(self.names)
        self.random_names = np.random.choice(self.names, 10, replace=False)

        if load_images is True:
            if DEBUG:
                images = np.zeros((10,height,width,3),dtype=np.float32)

                for n in range(len(self.random_names)):
                    images[n] = cv2.imread(EXTERNAL_DATA + IMAGE_FOLDER + self.random_names[n])
            else:
                images = np.zeros((self.len, height, width, 3), dtype=np.float32)
                for n in range(len(self.names)):
                    images[n] = cv2.imread(EXTERNAL_DATA + IMAGE_FOLDER + self.names[n])
        else :
            images = None


        if load_masks is True:
            if DEBUG:
                masks = np.zeros((10,height,width,3),dtype=np.float32)

                for n in range(len(self.random_names)):
                    masks[n] = cv2.imread(EXTERNAL_DATA + LABEL_FOLDER + self.random_names[n])
            else:
                masks = np.zeros((self.len, height, width, 3), dtype=np.float32)
                for n in range(len(self.names)):
                    masks[n] = cv2.imread(EXTERNAL_DATA + LABEL_FOLDER + self.names[n])
        else:
            masks = None

        self.images = images
        self.masks  = masks

    def imshow(self, imagePath='', add_mask=False ):
        if imagePath == '':
            index = np.random.choice(len(self.images), 1, replace=False)
            image = self.images[int(index)]
            if add_mask:
                mask = self.masks[int(index)]
                result = cv2.addWeighted(image, 0.5, mask, 0.5 , 0.)  # image * α + mask * β + λ
                plt.imshow(result.astype(np.uint8), interpolation='nearest')
                plt.waitforbuttonpress()
            else:
                plt.imshow(image.astype(np.uint8), interpolation='nearest')
                plt.waitforbuttonpress()

    def extract_objects(self):
        count = 0

        for n in range(len(self.names)):
            filename = self.names[n]
            mask = cv2.imread(EXTERNAL_DATA + LABEL_FOLDER + filename)
            image = cv2.imread(EXTERNAL_DATA + IMAGE_FOLDER + filename)

            for obj_class, color in OBJ_DICT.items():

                tmp_mask = (mask[:, :, :3] == color).all(2)
                tmp_mask = tmp_mask.astype(np.uint8)
                if (tmp_mask==1).any():
                    if len(tmp_mask[tmp_mask==1])/(IMAGE_HEIGHT*IMAGE_WIDTH)>0.1:
                        count += 1
                        tmp_image = tmp_mask[:, :, np.newaxis] * image
                        out_name = '{}_{:07d}.jpg'.format(obj_class, count)

                        cv2.imwrite(OUT_FOLDER + LABEL_FOLDER + out_name, tmp_mask * 255, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        cv2.imwrite(OUT_FOLDER + IMAGE_FOLDER + out_name, tmp_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

            print("\r Progress: %0.2f , Images: %d/%d" %(((n+1)/len(self.names))*100,(n+1), len(self.names)),flush=True,end='' )



    def debug_image(self):
        out_dir = EXTERNAL_DATA + '/car_labels/temp.jpg'
        mask = cv2.imread(EXTERNAL_DATA + LABEL_FOLDER + '00004.png')
        image = cv2.imread(EXTERNAL_DATA + IMAGE_FOLDER + '00004.png')


        mask = (mask[:, :, :3] == [142, 0, 0]).all(2)
        mask = mask.astype(np.uint8)

        result = mask[:,:,np.newaxis]*image

        # cv2.imwrite(out_dir, mask*255, [cv2.IMWRITE_JPEG_QUALITY, 100])

        plt.imshow(result)
        plt.waitforbuttonpress()



if __name__ == '__main__':
    pass
    # imgdb = ExternalData(load_images=False,load_masks=False)
    # imgdb.imshow(add_mask = True)
    # imgdb.extract_objects()