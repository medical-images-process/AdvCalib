import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from packaging import version

from model.deeplab import Res_Deeplab, ModelWithTemperature
from model.discriminator import FCDiscriminator
from utils.loss import * # CrossEntropy2d, BCEWithLogitsLoss2d, WeightedCE2d
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet



import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 2 # 10
ITER_SIZE = 5
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train.txt' # train_aug.txt'
DATA_LIST_PATH_REMAIN =  './dataset/voc_list/train_unlabeled.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321' # original image has 500,366
LEARNING_RATE = 2.5e-4 #
MOMENTUM = 0.9 #seg
NUM_CLASSES = 21
NUM_STEPS = 20000  #25000 , 1epoch (10582) = batch_size 10 * 1000 iterations, 20 epochs!
POWER = 0.9
RANDOM_SEED = 1234#  1234
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth' # resNet 101 COCO
# or pth.tar
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000 #5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4 # 1e-4
LAMBDA_ADV_PRED =0.01 #0.01 # lambda_adv for labeled 0.01

PARTIAL_DATA=0.125 ##

SEMI_START= 5000# 5000 #12500 #6250 # start semi after 5000 iterations, 6250 for 8
LAMBDA_SEMI=0.1  #lambda_semi
MASK_T=0.2 # mask threshold

LAMBDA_SEMI_ADV=0.001# 0.001 #lambda_adv for unlabeled
SEMI_START_ADV= 0
D_REMAIN= True
USED = True
USECALI = False


EPSILON = 10
ALPHA = 1
METHOD = ""


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.

      --snapshot-dir snapshots \
                --partial-data 0.125 \
                --num-steps 20000 \   25000 for 8
                --lambda-adv-pred 0.01 \
                --lambda-semi 0.1 --semi-start 5000 --mask-T 0.2
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-list-remain", type=str, default=DATA_LIST_PATH_REMAIN,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.") # use 1/8
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.") # save loss, ids, and etc...
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--epsilon", type=float, default=EPSILON,
                        help="epsilon.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="alpha")
    parser.add_argument("--method", type=str, default=METHOD,
                        help="vatent or just vat")

    return parser.parse_args()

args = get_arguments() #list of parsed data

def loss_calc(pred, label, gpu): # pred 8,21,321,321, label 8,1,321,321
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # pred shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def weighted_loss_calc(pred, label, gpu, confidence): # pred 8,21,321,321, label 8,321,321, confidence 8,321,321
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # pred shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    confidence = Variable(confidence.float()).cuda(gpu)
    criterion = WeightedCE2d().cuda(gpu)
    return criterion(pred,label,confidence)

def calibrated_loss_calc(pred, label, gpu, confidence,accuracies, n_bin): # pred 8,21,321,321, label 8,321,321, confidence 8,321,321
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # pred shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    confidence = Variable(confidence.float()).cuda(gpu)
    criterion = CalibratedCE2d().cuda(gpu)
    #criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred,label,confidence,accuracies,n_bin)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
# polynomial decreasing lr
## decreased with polynomial decay with power of 0.9 as mentioned in deeplab

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power) #lr_poly!
    #print(len(optimizer.param_groups)) >> 2


    optimizer.param_groups[0]['lr'] = lr # len(optimizer.param_groups) ==2
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10
# lr of Discriminator

def one_hot(label):
    label = label.numpy() # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w, 0~20 # 8*321*321 = batchsize*h*w
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype) # (8,21,321,321)
    for i in range(args.num_classes):# i = 0~20
        one_hot[:,i,...] = (label==i) # the value of entry is label!
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def make_D_label(label, ignore_mask): # 1 or 0 and 8,321,321
    ignore_mask = np.expand_dims(ignore_mask, axis=1) #8,1,321,321
    D_label = np.ones(ignore_mask.shape)*label # label 1 (GT) or 0 (pred)
    D_label[ignore_mask] = 255 #  since there are some regions are not labeled by the GT, we do not train the discriminator on these locations. 255 is just an arbitrary number for ignoring loss values.
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label

def make_conf_label(label, fake_mask): # label == 1 and (10*321*321, )
    conf_label = np.ones(fake_mask.shape)*label # label 1 (GT) or 0 (pred)
    conf_label[fake_mask] = 0 #  since there are some regions are not labeled by the GT, we do not train the discriminator on these locations. 255 is just an arbitrary number for ignoring loss values.
    conf_label = torch.FloatTensor(conf_label).cuda()

    return conf_label
#
# For some dataset (e.g., PASCAL), there are some boundary pixels are labeled as 255 (ignored).
# If your dataset doesn't have this scenario. It should be ok to directly apply the code

def main():
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network

    model = Res_Deeplab(num_classes=args.num_classes)

    # load pretrained parameters (weights)
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from) ## http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth
    else:
        saved_state_dict = torch.load(args.restore_from)
        #checkpoint = torch.load(args.restore_from)_

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy() # state_dict() is current model
    for name, param in new_params.items():
        #print (name) # 'conv1.weight, name:param(value), dict
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            #print('copy {}'.format(name))
    model.load_state_dict(new_params)
    #model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(args.checkpoint['optim_dict'])

    model.train() # https://pytorch.org/docs/stable/nn.html, Sets the module in training mode.
    model.cuda(args.gpu) ##

    cudnn.benchmark = True # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware

    # init D

    model_D = FCDiscriminator(num_classes=args.num_classes)
    #args.restore_from_D = 'snapshots/linear2/VOC_25000_D.pth'
    if args.restore_from_D is not None: # None
        model_D.load_state_dict(torch.load(args.restore_from_D))
        # checkpoint_D = torch.load(args.restore_from_D)
        # model_D.load_state_dict(checkpoint_D['state_dict'])
        # optimizer_D.load_state_dict(checkpoint_D['optim_dict'])
    model_D.train()
    model_D.cuda(args.gpu)

    if USECALI:
        model_cali = ModelWithTemperature(model, model_D)
        model_cali.cuda(args.gpu)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    train_dataset_remain = VOCDataSet(args.data_dir, args.data_list_remain, crop_size=input_size,
                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)
    train_dataset_size_remain = len(train_dataset_remain)

    print train_dataset_size
    print train_dataset_size_remain

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    if args.partial_data is None: #if not partial, load all

        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

        trainloader_gt = data.DataLoader(train_gt_dataset,
               batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    else:
        #sample partial data
        #args.partial_data = 0.125
        partial_size = int(args.partial_data * train_dataset_size)

        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id))
            print('loading train ids from {}'.format(args.partial_id))
        else: #args.partial_id is none
            train_ids = range(train_dataset_size)
            train_ids_remain = range(train_dataset_size_remain)
            np.random.shuffle(train_ids) #shuffle!
            np.random.shuffle(train_ids_remain)

        pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id.pkl'), 'wb')) #randomly suffled ids

        #sampler
        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:])  # 0~1/8,
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids_remain[:])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:])
        # train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size]) # 0~1/8
        # train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:]) # used as unlabeled, 7/8
        # train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        #train loader
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=3, pin_memory=True) # multi-process data loading
        trainloader_remain = data.DataLoader(train_dataset_remain,
                        batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=3, pin_memory=True)
        # trainloader_remain = data.DataLoader(train_dataset,
        #                                      batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=3,
        #                                     pin_memory=True)
        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=3, pin_memory=True)

        trainloader_remain_iter = enumerate(trainloader_remain)


    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network

    # model.optim_paramters(args) = list(dict1, dict2), dict1 >> 'lr' and 'params'
    # print(type(model.optim_parameters(args)[0]['params'])) # generator
    #print(model.state_dict()['coeff'][0]) #confirmed

    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    #optimizer.add_param_group({"params":model.coeff}) # assign new coefficient to the optimizer
    #print(len(optimizer.param_groups))
    optimizer.zero_grad()


    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad() #initialize

    if USECALI:
        optimizer_cali = optim.LBFGS([model_cali.temperature], lr=0.01, max_iter=50)
        optimizer_cali.zero_grad()

        nll_criterion = BCEWithLogitsLoss().cuda() # BCE!!
        ece_criterion = ECELoss().cuda()


    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear') # okay it automatically change to functional.interpolate
    # 321, 321

    if version.parse(torch.__version__) >= version.parse('0.4.0'):  #0.4.1
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')


    # labels for adversarial training
    pred_label = 0
    gt_label = 1
    semi_ratio_sum = 0
    semi_sum = 0
    loss_seg_sum = 0
    loss_adv_sum = 0
    loss_vat_sum = 0
    l_seg_sum =0
    l_vat_sum =0
    l_adv_sum = 0
    logits_list = []
    labels_list = []

    #https: // towardsdatascience.com / understanding - pytorch -with-an - example - a - step - by - step - tutorial - 81fc5f8c4e8e


    for i_iter in range(args.num_steps):

        loss_seg_value = 0 # L_seg
        loss_adv_pred_value = 0 # 0.01 L_adv
        loss_D_value = 0 # L_D
        loss_semi_value = 0 # 0.1 L_semi
        loss_semi_adv_value = 0 # 0.001 L_adv
        loss_vat_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter) #changing lr by iteration

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)



        for sub_i in range(args.iter_size):

            ###################### train G!!!###########################
            ############################################################
            # don't accumulate grads in D

            for param in model_D.parameters(): # <class 'torch.nn.parameter.Parameter'>, convolution weights
                param.requires_grad = False # do not update gradient of D (freeze) while G


            ######### do unlabeled first!! 0.001 L_adv + 0.1 L_semi ###############

            # lambda_semi, lambda_adv for unlabeled
            if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and i_iter >= args.semi_start_adv :
                try:
                    _, batch = trainloader_remain_iter.next() #remain = unlabeled
                    print(trainloader_remain_iter.next())
                except:
                    trainloader_remain_iter = enumerate(trainloader_remain) # impose counters
                    _, batch = trainloader_remain_iter.next()

                # only access to img
                images, _, _, _ = batch # <class 'torch.Tensor'>
                images = Variable(images).cuda(args.gpu) # <class 'torch.Tensor'>

                pred = interp(model(images)) # S(X), pred <class 'torch.Tensor'>
                pred_remain = pred.detach() #use detach() when attempting to remove a tensor from a computation graph, will be used for D
                # https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861
                # The difference is that detach refers to only a given variable on which it's called.
                # torch.no_grad affects all operations taking place within the with statement. >> for context,
                # requires_grad is for tensor

                # pred >> (8,21,321,321), L_adv
                D_out = interp(model_D(F.softmax(pred))) # D(S(X)), confidence, 8,1,321,321, not detached, there was not dim
                D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1) # (8,321,321) 0~1



                # 0.001 L_adv!!!!
                ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool) # no ignore_mask for unlabeled adv
                loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain)) #gt_label =1,
                # -log(D(S(X)))
                loss_semi_adv = loss_semi_adv/args.iter_size #normalization

                loss_semi_adv_value += loss_semi_adv.data.cpu().numpy() / args.lambda_semi_adv

                ##--- visualization, pred(8,21,321,321), D_out_sigmoid(8,321,321)
                """
                if i_iter % 1000 == 0:
                    vpred = pred.transpose(1, 2).transpose(2, 3).contiguous()  # (8,321,321,21)
                    vpred = vpred.view(-1, 21)  # (8*321*321, 21)
                    vlogsx = F.log_softmax(vpred)  # torch.Tensor
                    vsemi_gt = pred.data.cpu().numpy().argmax(axis=1)
                    vsemi_gt = Variable(torch.FloatTensor(vsemi_gt).long()).cuda(gpu)
                    vlogsx = vlogsx.gather(1, vsemi_gt.view(-1, 1))
                    sx = F.softmax(vpred).gather(1, vsemi_gt.view(-1, 1))
                    vD_out_sigmoid = Variable(torch.FloatTensor(D_out_sigmoid)).cuda(gpu).view(-1, 1)
                    vlogsx = (vlogsx*(2.5*vD_out_sigmoid+0.5))
                    vlogsx = -vlogsx.squeeze(dim=1)
                    sx = sx.squeeze(dim=1)
                    vD_out_sigmoid = vD_out_sigmoid.squeeze(dim=1)
                    dsx = vD_out_sigmoid.data.cpu().detach().numpy()
                    vlogsx = vlogsx.data.cpu().detach().numpy()
                    sx = sx.data.cpu().detach().numpy()
                    plt.clf()
                    plt.figure(figsize=(15, 5))
                    plt.subplot(131)
                    plt.ylim(0, 0.004)
                    plt.scatter(dsx, vlogsx, s = 0.1)  # variable requires grad cannot call numpy >> detach
                    plt.xlabel('D(S(X))')
                    plt.ylabel('Loss_Semi per Pixel')
                    plt.subplot(132)
                    plt.scatter(dsx, vlogsx, s = 0.1)  # variable requires grad cannot call numpy >> detach
                    plt.xlabel('D(S(X))')
                    plt.ylabel('Loss_Semi per Pixel')
                    plt.subplot(133)
                    plt.scatter(dsx, sx, s=0.1)
                    plt.xlabel('D(S(X))')
                    plt.ylabel('S(x)')
                    plt.savefig('/home/eungyo/AdvSemiSeg/plot/'  + str(i_iter) + '.png')
                    """

                if args.lambda_semi <= 0 or i_iter < args.semi_start:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:

                    semi_gt = pred.data.cpu().numpy().argmax(axis=1) # pred=S(X) ((8,21,321,321)), semi_gt is not one-hot, 8,321,321
                    #(8, 321, 321)

                    if not USECALI:
                        semi_ignore_mask = (D_out_sigmoid < args.mask_T)  # both (8,321,321) 0~1threshold!, numpy
                        semi_gt[semi_ignore_mask] = 255 # Yhat, ignore pixel becomes 255
                        semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size # ignored pixels / H*W
                        print('semi ratio: {:.4f}'.format(semi_ratio))

                        if semi_ratio == 0.0:
                            loss_semi_value += 0
                        else:
                            semi_gt = torch.FloatTensor(semi_gt)
                            confidence = torch.FloatTensor(D_out_sigmoid)  ## added, only pred is on cuda
                            loss_semi = args.lambda_semi * weighted_loss_calc(pred, semi_gt, args.gpu, confidence)

                    else:
                        semi_ratio =1
                        semi_gt = (torch.FloatTensor(semi_gt)) # (8,321,321)
                        confidence = torch.FloatTensor(F.sigmoid(model_cali.temperature_scale(D_out.view(-1))).data.cpu().numpy()) # (8*321*321,)
                        loss_semi = args.lambda_semi * calibrated_loss_calc(pred, semi_gt, args.gpu, confidence,accuracies, n_bin) #  L_semi = Yhat * log(S(X)) # loss_calc(pred, semi_gt, args.gpu)
                        # pred(8,21,321,321)

                    if semi_ratio !=0:
                        loss_semi = loss_semi/args.iter_size
                        loss_semi_value += loss_semi.data.cpu().numpy()/args.lambda_semi

                        if args.method == 'vatent' or args.method == 'vat':
                            #v_loss = vat_loss(model, images, pred, eps=args.epsilon[i])  # R_vadv
                            weighted_v_loss = weighted_vat_loss(model, images, pred, confidence, eps=args.epsilon)

                            if args.method == 'vatent':
                                #v_loss += entropy_loss(pred)  # R_cent (conditional entropy loss)
                                weighted_v_loss += weighted_entropy_loss(pred, confidence)


                            v_loss = weighted_v_loss / args.iter_size
                            loss_vat_value += v_loss.data.cpu().numpy()
                            loss_semi_adv += args.alpha * v_loss

                            loss_vat_sum += loss_vat_value
                            if i_iter % 100 == 0 and sub_i ==4:
                                l_vat_sum = loss_vat_sum / 100
                                if i_iter == 0:
                                    l_vat_sum = l_vat_sum * 100
                                loss_vat_sum = 0

                        loss_semi += loss_semi_adv
                        loss_semi.backward() # 0.001 L_adv + 0.1 L_semi, backward == back propagation

            else:
                loss_semi = None
                loss_semi_adv = None


            ###########train with source (labeled data)############### L_ce + 0.01 * L_adv

            try:
                _, batch = trainloader_iter.next()
            except:
                trainloader_iter = enumerate(trainloader) # safe coding
                _, batch = trainloader_iter.next() #counter, batch

            images, labels, _, _ = batch # also get labels images(8,321,321)
            images = Variable(images).cuda(args.gpu)
            ignore_mask = (labels.numpy() == 255) # ignored pixels == 255 >> 1, yes ignored mask for labeled data

            pred = interp(model(images)) # S(X), 8,21,321,321
            loss_seg = loss_calc(pred, labels, args.gpu) # -Y*logS(X)= L_ce, not detached

            if USED:
                softsx = F.softmax(pred,dim=1)
                D_out = interp(model_D(softsx)) # D(S(X)), L_adv

                loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask)) # both  8,1,321,321, gt_label = 1
                # L_adv =  -log(D(S(X)), make_D_label is all 1 except ignored_region

                loss = loss_seg + args.lambda_adv_pred * loss_adv_pred
                if USECALI:
                    if (args.lambda_semi > 0 or args.lambda_semi_adv > 0) and i_iter >= args.semi_start_adv:
                        with torch.no_grad():
                            _, prediction = torch.max(softsx, 1)
                            labels_mask = ((labels > 0) * (labels != 255)) | (prediction.data.cpu() > 0)
                            labels = labels[labels_mask]
                            prediction = prediction[labels_mask]
                            fake_mask = (labels.data.cpu().numpy() != prediction.data.cpu().numpy())
                            real_label = make_conf_label(1, fake_mask)  # (10*321*321, ) 0 or 1 (fake or real)

                            logits = D_out.squeeze(dim=1)
                            logits = logits[labels_mask]
                            logits_list.append(logits)  # initialize
                            labels_list.append(real_label)

                        if (i_iter*args.iter_size*args.batch_size + sub_i+1)%train_dataset_size ==0:
                            logits = torch.cat(logits_list).cuda()  # overall 5000 images in val,  #logits >> 5000,100, (1464*321*321,)
                            labels = torch.cat(labels_list).cuda()
                            before_temperature_nll = nll_criterion(logits, labels).item()  ####modify
                            before_temperature_ece, _, _= ece_criterion(logits, labels)  # (1464*321*321,)
                            before_temperature_ece = before_temperature_ece.item()
                            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

                            def eval():
                                loss_cali = nll_criterion(model_cali.temperature_scale(logits), labels)
                                loss_cali.backward()
                                return loss_cali

                            optimizer_cali.step(eval)  # just one backward >> not 50 iterations
                            after_temperature_nll = nll_criterion(model_cali.temperature_scale(logits), labels).item()
                            after_temperature_ece, accuracies, n_bin = ece_criterion(model_cali.temperature_scale(logits), labels)
                            after_temperature_ece = after_temperature_ece.item()
                            print('Optimal temperature: %.3f' % model_cali.temperature.item())
                            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

                            logits_list = []
                            labels_list = []



            else:
                loss = loss_seg

            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            loss_seg_sum += loss_seg/args.iter_size
            if USED:
                loss_adv_sum += loss_adv_pred
            if i_iter %100 ==0 and sub_i ==4:
                l_seg_sum = loss_seg_sum/100
                if USED:
                    l_adv_sum = loss_adv_sum/100
                if i_iter ==0:
                    l_seg_sum = l_seg_sum*100
                    l_adv_sum = l_adv_sum*100
                loss_seg_sum = 0
                loss_adv_sum = 0

            loss_seg_value += loss_seg.data.cpu().numpy()/args.iter_size
            if USED:
                loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()/args.iter_size


            ##################### train D!!!###########################
            ###########################################################
            # bring back requires_grad
            if USED:
                for param in model_D.parameters():
                    param.requires_grad = True  # before False.

                ############# train with pred S(X)############# labeled + unlabeled
                pred = pred.detach()  #orginally only use labeled data, freeze S(X) when train D,

                # We do train D with the unlabeled data. But the difference is quite small

                if args.D_remain: #default true
                    pred = torch.cat((pred, pred_remain), 0) # pred_remain(unlabeled S(x)) is detached  16,21,321,321
                    ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0) # 16,321,321

                D_out = interp(model_D(F.softmax(pred, dim=1))) # D(S(X)) 16,1,321,321  # softmax(pred,dim=1) for 0.4, not nessesary

                loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask)) # pred_label = 0

                # -log(1-D(S(X)))
                loss_D = loss_D/args.iter_size/2 # iter_size = 1, /2 because there is G and D
                loss_D.backward()
                loss_D_value += loss_D.data.cpu().numpy()



                ################## train with gt################### only labeled
                #VOCGT and VOCdataset can be reduced to one dataset in this repo.
                # get gt labels Y
                #print "before train gt"
                try:
                    print(trainloader_gt_iter.next())# len 732
                    _, batch = trainloader_gt_iter.next()

                except:
                    trainloader_gt_iter = enumerate(trainloader_gt)
                    _, batch = trainloader_gt_iter.next()
                #print "train with gt?"
                _, labels_gt, _, _ = batch
                D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu) #one_hot
                ignore_mask_gt = (labels_gt.numpy() == 255) # same as ignore_mask (8,321,321)
                #print "finish"
                D_out = interp(model_D(D_gt_v))  # D(Y)
                loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt)) # log(D(Y))
                loss_D = loss_D/args.iter_size/2

                loss_D.backward()
                loss_D_value += loss_D.data.cpu().numpy()



        optimizer.step()

        if USED:
            optimizer_D.step()


        print('exp = {}'.format(args.snapshot_dir)) #snapshot
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.6f}, loss_semi = {5:.6f}, loss_semi_adv = {6:.3f}, loss_vat = {7: .5f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value, loss_vat_value))
#                                        L_ce             L_adv for labeled      L_D                L_semi              L_adv for unlabeled
#loss_adv should be inversely proportional to the loss_D if they are seeing the same data.
# loss_adv_p is essentially the inverse loss of loss_D. We expect them to achieve a good balance during the adversarial training
# loss_D is around 0.2-0.5   >> good
        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            #torch.save(state, osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '.pth.tar'))
            #torch.save(state_D, osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '_D.pth.tar'))
            break

        if i_iter%100==0 and sub_i ==4:                                                                                                                                                                                       #loss_seg_value
            wdata = "iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.6f}, loss_semi = {5:.8f}, loss_semi_adv = {6:.3f}, l_vat_sum = {7: .5f}, loss_label = {8: .4}\n".format(i_iter, args.num_steps, l_seg_sum, l_adv_sum, loss_D_value, loss_semi_value, loss_semi_adv_value, l_vat_sum, l_seg_sum+0.01*l_adv_sum)
            #wdata2 = "{0:8d} {1:s} {2:s} {3:s} {4:s} {5:s} {6:s} {7:s} {8:s}\n".format(i_iter,str(model.coeff[0])[8:14],str(model.coeff[1])[8:14],str(model.coeff[2])[8:14],str(model.coeff[3])[8:14],str(model.coeff[4])[8:14],str(model.coeff[5])[8:14],str(model.coeff[6])[8:14],str(model.coeff[7])[8:14])
            if i_iter==0:
                f2 = open("/home/eungyo/AdvSemiSeg/snapshots/log.txt", 'w')
                f2.write(wdata)
                f2.close()
                #f3 = open("/home/eungyo/AdvSemiSeg/snapshots/coeff.txt", 'w')
                #f3.write(wdata2)
                #f3.close()
            else:
                f1 = open("/home/eungyo/AdvSemiSeg/snapshots/log.txt", 'a')
                f1.write(wdata)
                f1.close()
                #f4 = open("/home/eungyo/AdvSemiSeg/snapshots/coeff.txt", 'a')
                #f4.write(wdata2)
                #f4.close()

        if i_iter % args.save_pred_every == 0 and i_iter!=0: # 5000
            print('taking snapshot ...')
            #state = {'epoch':i_iter, 'state_dict':model.state_dict(),'optim_dict':optimizer.state_dict()}
            #state_D = {'epoch':i_iter, 'state_dict': model_D.state_dict(), 'optim_dict': optimizer_D.state_dict()}
            #torch.save(state, osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '.pth.tar'))
            #torch.save(state_D, osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '_D.pth.tar'))
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '_D.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    # __name__ is __main__ when it is compiled directly, not imported
    main()
# BCE loss https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
# CE loss https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss

#torch.nn.functional.binary_cross_entropy takes logistic sigmoid values as inputs
#torch.nn.functional.binary_cross_entropy_with_logits takes logits as inputs (input can be before logistic)
#torch.nn.functional.cross_entropy takes logits as inputs (performs log_softmax internally)
# torch.nn.functional.nll_loss is like cross_entropy but takes log-probabilities (log-softmax)
