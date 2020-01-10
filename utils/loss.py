import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

interp = nn.Upsample(size=(321, 321), mode='bilinear', align_corners=True)
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #print predict.data.cpu().numpy().argmax(axis=1)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class WeightedCE2d(nn.Module):
    # https: // discuss.pytorch.org / t / weighted - pixelwise - nllloss2d / 7766
    def __init__(self, size_average=True, ignore_label=255):
        super(WeightedCE2d, self).__init__() # for python 2.x,  super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, confidence, weight=None): # forward(self, predict, target, weight=None):
        """
            pred 8,21,321,321, target 8,321,321 confidence 8,321,321, accuracies = list[15]
        """
        assert not target.requires_grad # target.requires_grad should be false
        assert predict.dim() == 4 # <class 'torch.Tensor'>
        assert target.dim() == 3
        assert confidence.dim() == 3 ### added
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        # treat ignore label!!
        n, c, h, w = predict.size()  #8,1,321,231 weight
                                                                    # target 8,321,321
        target_mask = (target >= 0) * (target != self.ignore_label) # target_mask(8,321,321), ignore_label== conf<0.2, True or False
        target = target[target_mask] #  target (824328,), only not ignored pixel can be survive torch.cuda.LongTensor
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous() #  >> to make (8, 321, 321, 21),
        # (8, 321, 321, 1), # 8,321,321,21 for repeat (copy the mask for 21 times)
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c) #  (17310888,) >> (824328, 21) by view, view need contigous tensor, transpose make tensor discontiguous,
        confidence = confidence[target_mask] # 824328,
        logp = F.log_softmax(predict) # (824328,21) logS(X), log_softmax
        ## Gather log probabilities with respect to target, logS(X) for only the right class
        logp = logp.gather(1, target.view(-1,1)) #(824328,) 1 for dim, target.view for index #index >> 0~20, indexth channel at dim 1
        ## Multiply with weights# 824328,1
        weighted_logp = logp * (50 / 32 * (confidence - 0.2) ** 2 + 1)
        weighted_loss = -weighted_logp.sum(0) / confidence.size(0)  # checked
        ## Average over mini-batch
        weighted_loss = weighted_loss[0]

        return weighted_loss
    #https: // discuss.pytorch.org / t / weighted - pixelwise - nllloss2d / 7766 # target will become one-hot by the paper
    # logsoftmax + nll_los (negative log likelihood), nll_los (softmax(predict),target)

class CalibratedCE2d(nn.Module):
    # https: // discuss.pytorch.org / t / weighted - pixelwise - nllloss2d / 7766
    def __init__(self, size_average=True, ignore_label=255):
        super(CalibratedCE2d, self).__init__() # for python 2.x,  super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, confidence, accuracies, n_bin, weight=None): # forward(self, predict, target, weight=None):
        """
            pred 8,21,321,321, target 8,321,321 confidence 8*321*321, accuracies = list[n_bin]
        """
        assert not target.requires_grad # target.requires_grad should be false
        assert predict.dim() == 4 # <class 'torch.Tensor'>
        assert target.dim() == 3
        assert confidence.dim() == 1
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()  #8,1,321,231 weight

        predict = predict.transpose(1, 2).transpose(2, 3).contiguous() #  >> to make (8, 321, 321, 21), dimension (1,2) and(2,3) will be transposed
        predict = predict.view(-1, c) # (824328, 21) by view, view need contigous tensor, transpose make tensor discontiguous,
        logp = F.log_softmax(predict) # (824328,21) logS(X), log_softmax
        ## Gather log probabilities with respect to target, logS(X) for only the right class
        #print (target >=40).sum(0)
        logp = logp.gather(1, target.view(-1,1)) #(824328,) 1 for dim, target.view for index #index >> 0~20, indexth channel at dim 1

        ## Multiply with weights
        bin_boundaries = torch.linspace(0, 1, 15+1)
        bin_lowers = bin_boundaries[:n_bin]
        bin_uppers = bin_boundaries[1:n_bin+1]
        bin_uppers[-1] = 1

        calibrated_loss = 0
        size = 0
        i=0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidence.gt(bin_lower.item()) * confidence.le(bin_upper.item())
            coeff = accuracies[i]*10 - (1-accuracies[i])*50 # * lambda.semi
            i+=1
            if coeff >0:
                size += accuracies[in_bin].size(0)
                weighted_logp = logp[in_bin] * coeff # check i~
                calibrated_loss -=weighted_logp.sum(0)

        calibrated_loss = calibrated_loss[0]/size

        return calibrated_loss


class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss






class BCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)   >> n*1*h*w
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 1
        assert target.dim() == 1
        assert predict.size() == target.size(), "{0} vs {1} ".format(predict.size(), target.size())

        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss



class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric)!!!!!

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin!!!

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):   # (1464*321*321, ) originally(256,100)  (256,)
        confidences = F.sigmoid(logits) # (1464*321*321,)  softmaxes = F.softmax(logits, dim=1)  # 5000,100
        # confidences(prob.) prediction (scalar)
        # predictions = (confidences >= 0.5).float()
        # confidences, predictions = torch.max(softmaxes, 1) (5000,)
        accuracies = labels # (1464*321*321,) !!!!!  S(x) == label, pixel accuracy

        acc_list =[]
        conf_list = []
        ece = torch.zeros(1, device=logits.device)
        # acc=0
        # len = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()) # 0 < and < 0.xx, (5000,)?, index
            prop_in_bin = in_bin.float().mean() #  # of indices  / 5000
            if prop_in_bin.item() > 0:
                # if bin_lower >= 0.2:
                #     acc += accuracies[in_bin].float().sum()
                #     len += (accuracies[in_bin]).size(0)
                accuracy_in_bin = accuracies[in_bin].float().mean()  #   /# of indices (mean), same in the paper
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin #  #of indices / (5000 * # of indices)

                acc_list.append(accuracy_in_bin.item())
                conf_list.append(avg_confidence_in_bin.item())

        print("acc",acc_list)
        # print(acc/len)
        print("conf", conf_list)

        return ece, acc_list, len(conf_list)






def kl_div_with_logit(q_logit, p_logit): # 8,21  vs 4,21,321,321

    n, c, h, w = q_logit.size()
    rest = h *w


    q = F.softmax(q_logit, dim=1)  # 8,21   vs 4,21,321,321
    logq = F.log_softmax(q_logit, dim=1) #
    logp = F.log_softmax(p_logit, dim=1)


    KLdv = (q * logq - q*logp).view(n, -1).sum(dim=1).mean(dim=0)/rest #
    # 4 remainder >> 4, >> ()
    #KLdv2 = (q * logq - q*logp ).view(n,c,-1).sum(dim=2).sum(dim=1).mean(dim=0)/(h*w)
    #KLdv3 = (q*logq - q*logp).sum(dim=1).sum(dim=1).sum(dim=1).mean(dim=0)/(h*w)

    #qlogq = ( q *logq).sum(dim=1).mean(dim=0)  #   128,10 > class_sum > (8,) > batch mean> ()
    #qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return KLdv

# diff from CE because, this time q is not labeled
# -qlogp >> CE, qloq >> detached (constant) >> extreme values is more important (larger loss)

def weighted_kl_div_with_logit(q_logit, p_logit, confidence): # 2,5,321,321
    n, c, h, w = q_logit.size()
    rest = h *w
    # confidence 2,321,321
    accept_mask = (confidence >= 0.2) # 2,321,321
    accept_mask = accept_mask.view(n,-1,h,w)
    accept_mask = accept_mask.repeat(1,c,1,1) #.float().cuda()

    q = F.softmax(q_logit, dim=1)  #  2,5,321,321
    logq = F.log_softmax(q_logit, dim=1) #
    logp = F.log_softmax(p_logit, dim=1)

    #KLdv = (q * logq - q * logp).view(n, -1).sum(dim=1).mean(dim=0) / rest
    weighted_KLdv = (q*logq-q*logp)[accept_mask].sum(dim=0)/n/rest #
     #
    # 4 remainder >> 4, >> ()
    #KLdv2 = (q * logq - q*logp ).view(n,c,-1).sum(dim=2).sum(dim=1).mean(dim=0)/(h*w)
    #KLdv3 = (q*logq - q*logp).sum(dim=1).sum(dim=1).sum(dim=1).mean(dim=0)/(h*w)

    #qlogq = ( q *logq).sum(dim=1).mean(dim=0)  #   128,10 > class_sum > (8,) > batch mean> ()
    #qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return weighted_KLdv


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16) #
    return torch.from_numpy(d)
    # 1e-16 for zero vector


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1): # eps 8 or 10
    # ul_x == images (4,3,321,321) right
    # ul_y == 4,21,321,321
    # find r_adv

    # 128, 3, 32, 32 ul_x
    # 128,10 ul_y (10 classes)

    d = torch.Tensor(ul_x.size()).normal_() # random tensor which has same size of x

    for i in range(num_iters):
        d = xi *_l2_normalize(d) # d is randomly sampled "unit" vector, at r= xi * d
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = interp(model(ul_x + d)) # p(y|x+r, theta), y_hat >> 8,21,321,321
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat) # ul_y = model(ul_x), delta_kl = D(r,x,theta)
        # detach because gradient is only for r
        delta_kl.backward()
        d = d.grad.data.clone().cpu()  # g = gradient_r D
        model.zero_grad()

    d = _l2_normalize(d) # g/||g||2
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = interp(model(ul_x + r_adv.detach()))
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)

    return delta_kl

def weighted_vat_loss(model, ul_x, ul_y, confidence, xi=1e-6, eps=2.5, num_iters=1): # eps 8 or 10
    # ul_x == images (4,3,321,321) right
    # ul_y == 4,21,321,321
    # find r_adv

    # 128, 3, 32, 32 ul_x
    # 128,10 ul_y (10 classes)

    d = torch.Tensor(ul_x.size()).normal_() # random tensor which has same size of x

    for i in range(num_iters):
        d = xi *_l2_normalize(d) # d is randomly sampled "unit" vector, at r= xi * d
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = interp(model(ul_x + d)) # p(y|x+r, theta), y_hat >> 8,21,321,321
        delta_kl = weighted_kl_div_with_logit(ul_y.detach(), y_hat,confidence) # ul_y = model(ul_x), delta_kl = D(r,x,theta)
        # detach because gradient is only for r
        delta_kl.backward()
        d = d.grad.data.clone().cpu()  # g = gradient_r D
        model.zero_grad()

    d = _l2_normalize(d) # g/||g||2
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = interp(model(ul_x + r_adv.detach()))
    delta_kl = weighted_kl_div_with_logit(ul_y.detach(), y_hat, confidence)

    return delta_kl


def entropy_loss(ul_y):
    n, c, h, w = ul_y.size()
    p = F.softmax(ul_y, dim=1) # p(y|x,theta), y for specific x,theta
    # exaggerating the prediction (ul_y) on each data point
    # 8,21  vs 8,21,321,321

    return -(p*F.log_softmax(ul_y, dim=1)).view(n, -1).sum(dim=1).mean(dim=0) / (h * w) #.sum(dim=1).mean(dim=0)

def weighted_entropy_loss(ul_y, confidence):
    n, c, h, w = ul_y.size()
    p = F.softmax(ul_y, dim=1) # p(y|x,theta), y for specific x,theta
    # exaggerating the prediction (ul_y) on each data point
    # 8,21  vs 8,21,321,321
    rest = h *w
    # confidence 2,321,321
    accept_mask = (confidence >= 0.2) # 2,321,321
    accept_mask = accept_mask.view(n,-1,h,w) # 2 1 321 321
    accept_mask = accept_mask.repeat(1,c,1,1) #.float().cuda()

    return -(p * F.log_softmax(ul_y, dim=1))[accept_mask].sum(dim=0)/n/rest #.sum(dim=1).mean(dim=0)
