from net.common import *
from net.dataset.dataprocess import *
from net.dataset.mask import *

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

def ensemble():

    model_dict={}

    # =============== Define the models to ensemble ===============

    model_dict['unet-py-8-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-1024/snap/final.pth",
        "UNet_pyramid_1024_2",
        1024,
    ]

    model_dict['unet-py-8-2-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-2-1024/snap/final.pth",
        "UNet_pyramid_1024_2",
        1024,
    ]

    # model_dict['unet-py-8-3-1024'] = [
    #     "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-3-1024/snap/final.pth",
    #     "UNet_pyramid_1024_2",
    #     1024,
    # ]

    model_dict['unet-py-8-4-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-4-1024/snap/final.pth",
        "UNet_pyramid_1024_2",
        1024,
    ]

    model_dict['unet-py-8-5-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-8-5-1024/snap/final.pth",
        "UNet_pyramid_1024_2",
        1024,
    ]

    model_dict['unet-6-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-6-1024/snap/gbest-9964.pth",
        "UNet512_shallow",
        1024,
    ]

    model_dict['unet-7-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-7-1024/snap/gbest-9962.pth",
        "UNet512_shallow",
        1024,
    ]

    model_dict['unet-py-0-1024'] = [
        "/home/eugene/Documents/Kaggle_Carvana/results/unet-py-0-1024/snap/final-9965.pth",
        "UNet_pyramid_1024",
        640,
    ]

    # ==============================================================

    # Define net
    model_dict = define_net(model_dict)
    predict_and_ensemble(model_dict, 1000 , "/home/eugene/Documents/Kaggle_Carvana/results/ensemble1")


def predict_and_ensemble(model_dict, block_size=CSV_BLOCK_SIZE, out_dir=None):
    gz_file = out_dir + "/results-ensemble.csv.gz"
    batch_size = 4
    rles       = []

    model_dict, test_num, names = define_test_dataset(model_dict, batch_size)


    assert (block_size % batch_size == 0)
    # assert (log != None)

    start0 = timer()

    num = 0
    for n in range(0, test_num, block_size):
        M = block_size if n + block_size < test_num else test_num - n
        print('[n=%d, M=%d]  \n' % (n, M))

        start = timer()

        ps = np.zeros((M, CARVANA_HEIGHT, CARVANA_WIDTH), np.uint16)
        for m in range(0, M, batch_size):
            print('\r\t%05d/%05d' % (m, M), end='', flush=True)

            # images, indices = test_iter.next()
            # if images is None:
            #     break

            for model_key, model_property in model_dict.items():
                images, indices = model_property[4].next()
                probs = predict(
                    model_path = model_property[0],
                    net = model_property[3] ,
                    images = images
                )
                batch_size = len(indices)
                ps[m: m + batch_size] = ps[m: m + batch_size] + probs

            num += batch_size

        for i in range(0, M):
            ps[i] = ps[i] / len(model_dict)
            pred = ps[i] > THRESHOLD

            # im_show('mask', pred, resize=0.25, cmap='gray')
            # plt.waitforbuttonpress()

            rle = run_length_encode(pred)
            rles.append(rle)

        print("Time: %2.2f min" %((timer()-start)/60))

    assert (test_num == num)
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')


def predict(model_path , net , images):

    # load model in net
    net.load_state_dict(torch.load(model_path))
    net.cuda().eval()

    # forward
    images = Variable(images, volatile=True).cuda()
    logits = net(images)
    probs = F.sigmoid(logits)

    if len(probs.size())<4:
        probs = torch.unsqueeze(probs, dim=1)
    probs = F.upsample(probs, (CARVANA_HEIGHT, CARVANA_WIDTH), mode="bilinear")
    probs = torch.squeeze(probs, dim=1)
    probs = probs.data.cpu().numpy() * 255

    return probs


def define_net(model_dict):
    for model_key,model_property in model_dict.items():
        if model_property[1]=="UNet_pyramid_1024_2":
            net = UNet_pyramid_1024_2(in_shape=(3, model_property[2], model_property[2]))
            model_dict[model_key].append(net)
        elif model_property[1]=="UNet512_shallow":
            net = UNet512_shallow(in_shape=(3, model_property[2], model_property[2]))
            model_dict[model_key].append(net)
        elif model_property[1]=="UNet_pyramid_1024":
            net = UNet_pyramid_1024(in_shape=(3, model_property[2], model_property[2]))
            model_dict[model_key].append(net)
        else:
            raise RuntimeError('No available model')
    return model_dict

def define_test_dataset(model_dict, batch_size=4):

    names = []

    for model_key, model_property in model_dict.items():
        if model_property[2]==1024:
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
            model_dict[model_key].append(test_iter)
        elif model_property[2]==640:
            test_dataset = ImageDataset('test-hq-100064',
                                        type='test',
                                        folder='test-INTER_LINEAR-640x640-hq',
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
            model_dict[model_key].append(test_iter)


    test_num = len(test_dataset)

    for n in range(0, test_num):
        name = test_dataset.names[n].split('/')[-1]
        name = name.replace('<mask>', '').replace('<ext>', 'jpg').replace('png', 'jpg')
        names.append(name)


    return model_dict, test_num, names




if __name__ == "__main__":

    ensemble()