
from net.common import *

KAGGLE_DATA_DIR ='/home/eugene/Documents/Kaggle_Carvana/data'

def run_remove_bg():

    img_dir  = KAGGLE_DATA_DIR + '/image/train-jpg'
    mask_dir = KAGGLE_DATA_DIR + '/image/mask-png'  # read all annotations

    save_dir  = KAGGLE_DATA_DIR + '/image/train-jpg-nbg'
    os.makedirs(save_dir,exist_ok=True)


    img_list = os.listdir(mask_dir)
    num_imgs = len(img_list)

    for n in range(num_imgs):
        img_file = img_list[n]
        shortname = img_file.split('.')[0].replace('_mask','')

        img_file = img_dir + '/%s.jpg'%(shortname)
        img = cv2.imread(img_file)

        mask_file = mask_dir + '/%s_mask.png'%(shortname)
        mask = cv2.imread(mask_file)/255.
        mask = np.array(mask[:,:,0])

        img = img*(np.dstack((mask,mask,mask)))
        cv2.imwrite(save_dir + '/%s.jpg'%shortname, img)


if __name__ == '__main__':
    pass
    # run_remove_bg()

