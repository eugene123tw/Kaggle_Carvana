from net.common import *
from net.model.loss import *
from net.dataset.dataprocess import *
from net.dataset.mask import *
from net.rates import *
from net.util import *


#decode and check
def run_csv_to_mask():

    gz_file = '/home/eugene/Documents/Kaggle_Carvana/results/unet-7-1024/submit/results-gbest-9962.csv.gz'
    out_dir = '/home/eugene/Documents/Kaggle_Carvana/results/unet-7-1024/pred_mask/'
    df = pd.read_csv(gz_file, compression='gzip', header=0)

    for n in range(len(df)):
        name = df.values[n][0]
        rle   = df.values[n][1]
        mask  = run_length_decode(rle,H=CARVANA_HEIGHT, W=CARVANA_WIDTH)
        # im_show('mask', mask, resize=0.25, cmap='gray')
        # plt.waitforbuttonpress()
        filename = out_dir + name.split('.')[0]+ '<pred>.' + name.split('.')[1]
        cv2.imwrite(filename, mask, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ =='__main__':
    run_csv_to_mask()