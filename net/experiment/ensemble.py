from net.common import *
from net.dataset.dataprocess import *

from net.model.unet import UNet512_2
from net.model.unet import UNet512_shallow
from net.model.unet import UNet1024
from net.model.networks import SegNet
from net.model.unet import UNet_pyramid_1
from net.model.unet import UNet_pyramid_1024
from net.model.unet import UNet_pyramid_1024_2


CARVANA_HEIGHT = 1280
CARVANA_WIDTH  = 1918
CSV_BLOCK_SIZE = 1080
THRESHOLD      = 127

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

def predict8_in_blocks(net, test_loader, block_size=CSV_BLOCK_SIZE, log=None, save_dir=None):

    test_dataset = test_loader.dataset
    test_iter    = iter(test_loader)
    test_num     = len(test_dataset)
    batch_size   = test_loader.batch_size
    assert(block_size%batch_size==0)
    assert(log!=None)

    start0 = timer()
    num  = 0
    predictions = []
    for n in range(0, test_num, block_size):
        M = block_size if n+block_size < test_num else test_num-n
        log.write('[n=%d, M=%d]  \n'%(n,M) )

        start = timer()

        ps  = None
        for m in range(0, M, batch_size):
            print('\r\t%05d/%05d'%(m,M), end='',flush=True)

            images, indices  = test_iter.next()
            if images is None:
                break


            # forward
            images = Variable(images,volatile=True).cuda()
            logits = net(images)
            probs  = F.sigmoid(logits)

            #save results
            if ps is None:
                H = images.size(2)
                W = images.size(3)
                ps  = np.zeros((M, H, W),np.uint8)
                ids = np.zeros((M),np.int64)


            batch_size = len(indices)
            indices = indices.cpu().numpy()
            probs = probs.data.cpu().numpy() *255

            ps [m : m+batch_size] = probs
            ids[m : m+batch_size] = indices
            num  += batch_size
            # im_show('probs',probs[0],1)
            # cv2.waitKey(0)
            #print('\tm=%d, m+batch_size=%d'%(m,m+batch_size) )

        pass # end of one block -----------------------------------------------------
        print('\r')
        log.write('\tpredict = %0.2f min, '%((timer() - start) / 60))


        ##if(n<64000): continue
        if save_dir is not None:
            start = timer()
            np.savetxt(save_dir+'/indices-part%02d.8.txt'%(n//block_size), ids, fmt='%d')
            np.save(save_dir+'/probs-part%02d.8.npy'%(n//block_size), ps) #  1min
            log.write('save = %0.2f min'%((timer() - start) / 60))
        log.write('\n')


    log.write('\nall time = %0.2f min\n'%((timer() - start0) / 60))
    assert(test_num == num)


def predict_and_ensemble(model_dict, block_size=CSV_BLOCK_SIZE, out_dir=None):
    gz_file = out_dir + "/results-ensemble.csv.gz"
    batch_size = 4
    rles       = []
    names      = []


    test_dataset = ImageDataset('test-hq-100064',
                                type='test',
                                folder='test-INTER_LINEAR-1024x1024-hq',
                                )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
        pin_memory=True)

    test_dataset = test_loader.dataset
    test_iter = iter(test_loader)
    test_num = len(test_dataset)


    for n in range(0, test_num):
        name = test_dataset.names[n].split('/')[-1]
        name = name.replace('<mask>', '').replace('<ext>', 'jpg').replace('png', 'jpg')
        names.append(name)


    assert (block_size % batch_size == 0)
    # assert (log != None)

    start0 = timer()

    num = 0
    for n in range(0, test_num, block_size):                          # 0, 1000, 2000, ..., 100064
        M = block_size if n + block_size < test_num else test_num - n # M = 1000
        # log.write('[n=%d, M=%d]  \n' % (n, M))

        start = timer()

        ps = None
        for m in range(0, M, batch_size):                             # 0, 4, 8, 12, ...., 1000
            print('\r\t%05d/%05d' % (m, M), end='', flush=True)

            images, indices = test_iter.next()
            if images is None:
                break

            for model_name, model_property in model_dict.items():

                model_path = model_property[0]
                net = model_property[3]
                net.load_state_dict(torch.load(model_path))
                net.cuda().eval()


                # forward
                tmp_images = Variable(images, volatile=True).cuda()
                logits = net(tmp_images)
                probs = F.sigmoid(logits)
                probs = torch.unsqueeze(probs, dim=1)
                probs = F.upsample(probs, (CARVANA_HEIGHT, CARVANA_WIDTH), mode="bilinear")
                probs = torch.squeeze(probs, dim=1)
                # save results
                if ps is None:
                    ps = np.zeros((M, CARVANA_HEIGHT, CARVANA_WIDTH), np.uint8)
                    # ids = np.zeros((M), np.int64)

                batch_size = len(indices)
                probs = probs.data.cpu().numpy() * 255
                ps[m: m + batch_size] = ps[m: m + batch_size] + probs

            num += batch_size

        ps = ps/len(model_dict)

        for n in range(0, M):
            pred = ps[n] > THRESHOLD
            rle = run_length_encode(pred)
            rles.append(rle)

    assert (test_num == num)
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')



def im_show(name, image, resize=1, cmap = ''):
    H,W = image.shape[0:2]
    if cmap=='gray':
        plt.imshow(image.astype(np.uint8),cmap='gray')
    plt.imshow(image.astype(np.uint8))


#https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def run_length_decode(rel, H, W, fill_value=255):
    mask = np.zeros((H * W), np.uint8)
    rel = np.array([int(s) for s in rel.split(' ')]).reshape(-1, 2)
    for r in rel:
        start = r[0]
        end = start + r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(H, W)
    return mask


def ensemble_csv(csv_list, gz_file):
    rles = []
    names = []

    for n in range(0,100064):
        accu_mask = 0
        for file_indices in range(0, len(csv_list)):
            df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0, skiprows= n, nrows = 1)
            rle = df.values[0][1]
            accu_mask += run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH)/255

        names.append(df.values[0][0]) # Append name
        accu_mask = accu_mask > len(csv_list)/2

        # im_show('mask', accu_mask, resize=0.25, cmap='gray')
        # plt.waitforbuttonpress()

        rle = run_length_encode(accu_mask)
        rles.append(rle)
        print('rle : b/num_test = %06d/%06d' % (n, 100064))


    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


def ensemble_csv_v2(csv_list, out_dir):

    gz_file = out_dir + "/results-ensemble.csv.gz"

    rles = []
    names = []

    num = 100064
    H, W = CARVANA_HEIGHT, CARVANA_WIDTH
    binary_pred = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(num, H, W))

    print("Do merge CSV")
    for file_indices in range(0, len(csv_list)):
        df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0)
        for n in range(0, num):
            rle = df.values[n][1]
            binary_pred[n] = binary_pred[n] + (run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH)/255)
            if file_indices==0:
                names.append(df.values[n][0])
            if n%1000==0:
                print('File indices: %06d/%06d, rle : b/num_test = %06d/%06d' %
                      (file_indices, len(csv_list), n + 1, num)
                )

    binary_pred = binary_pred > (len(csv_list)/2)

    print("Do run_length_encode")
    for n in range(0, num):
        mask = binary_pred[n]
        rle = run_length_encode(mask)
        rles.append(rle)
        print('rle : b/num_test = %06d/%06d' % (n+1, 100064))

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


def ensemble_csv_v3(csv_list, out_dir, chunk_size=1000.0):

    gz_file = out_dir + "/results-ensemble.csv.gz"

    rles = []
    names = []

    num = 100064
    H, W = CARVANA_HEIGHT, CARVANA_WIDTH
    chunk_batch = int(math.ceil(num/chunk_size)) # 100064/1000. = 100.064

    begin, end = 0, 0

    for chunk in range(0, chunk_batch):
        begin = int(end)                 # 2000
        end = int(begin + chunk_size)    # 3000
        if end > num : end = int(num)

        binary_pred = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(end-begin, H, W))
        for file_indices in range(0, len(csv_list)):
            df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0) # 0 - 100064
            for n in range(begin, end):  # 2000-3000
                rle = df.values[n][1]
                binary_pred[n-begin] = binary_pred[n-begin] + (run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH) / 255)
                if file_indices == 0:
                    names.append(df.values[n][0])
                if n % 100 == 0:
                    print('File indices: %02d/%02d, rle : b/num_test = %06d/%06d' %
                          (file_indices+1, len(csv_list), n + 1, num)
                          )

        binary_pred = binary_pred > (len(csv_list) / 2)
        print("Do run_length_encode")
        for n in range(begin, end):
            mask = binary_pred[n-begin]
            rle = run_length_encode(mask)
            rles.append(rle)
            if n%100==0:
                print('rle : b/num_test = %06d/%06d' % (n + 1, num))

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


def ensemble():

    model_dict={}

    # =============== Define the models to ensemble ===============
    model_dict['unet-py-8-2-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-2-1024/snap/final.pth",
        "UNet_pyramid_1024_2",
        1024,
    ]

    model_dict['unet-py-8-3-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-3-1024/snap/final.pth",
        "UNet_pyramid_1024_2",
        1024,
    ]

    # model_dict['unet-6-1024'] = [
    #     "/Users/Eugene/Documents/Git/unet-6-1024/snap/final.pth",
    #     "UNet512_shallow",
    #     1024,
    # ]
    #
    # model_dict['unet-7-1024'] = [
    #     "/Users/Eugene/Documents/Git/unet-7-1024/snap/final.pth",
    #     "UNet512_shallow",
    #     1024,
    # ]

    # model_dict['unet-py-0-1024'] = [
    #     "/Users/Eugene/Documents/Git/unet-py-0-1024/snap/final.pth",
    #     "UNet_pyramid_1024",
    #     640,
    # ]

    # ==============================================================

    # Define net
    for key,val in model_dict.items():
        if val[1]=="UNet_pyramid_1024_2":
            net = UNet_pyramid_1024_2(in_shape=(3, val[2], val[2]))
            model_dict[key].append(net)
        elif val[1]=="UNet512_shallow":
            net = UNet512_shallow(in_shape=(3, val[2], val[2]))
            model_dict[key].append(net)
        elif val[1]=="UNet_pyramid_1024":
            net = UNet_pyramid_1024(in_shape=(3, val[2], val[2]))
            model_dict[key].append(net)
        else:
            raise RuntimeError('No available model')

    predict_and_ensemble(model_dict, 1000 , "/home/eugene/Documents/Kaggle_Carvana/results/ensemble")



if __name__ == "__main__":

    csv_list = [
        "/Users/Eugene/Documents/Git/fold1/results-final.csv.gz",
        "/Users/Eugene/Documents/Git/fold2/results-final.csv.gz",
    ]

    csv_list = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-1024/submit/results-final.csv.gz",
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-2-1024/submit/results-final.csv.gz",
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-3-1024/submit/results-final.csv.gz",
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-4-1024/submit/results-final.csv.gz",
    ]


    # ensemble_csv(csv_list,"/Users/Eugene/Documents/Git/results-ensemble.csv.gz")
    # ensemble_csv_v2(csv_list, "/Users/Eugene/Documents/Git")
    # ensemble_csv_v3(csv_list, "/Users/Eugene/Documents/Git", 1000.0)
    # ensemble_csv_v3(csv_list, "/home/eugene/Documents/Kaggle_Carvana/results/ensemble", 1000.0)
    ensemble()