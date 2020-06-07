import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import ramps



class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """
    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    return F.cross_entropy(input_logits/temperature, target_targets, ignore_index=ignore_index)


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """
    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                        reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type
        
        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1/num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh =  self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')



def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)
    
    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets+epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5



def pair_wise_loss(unsup_outputs, size_average=True, nbr_of_pairs=8):
	"""
	Pair-wise loss in the sup. mat.
	"""
	if isinstance(unsup_outputs, list):
		unsup_outputs = torch.stack(unsup_outputs)

	# Only for a subset of the aux outputs to reduce computation and memory
	unsup_outputs = unsup_outputs[torch.randperm(unsup_outputs.size(0))]
	unsup_outputs = unsup_outputs[:nbr_of_pairs]

	temp = torch.zeros_like(unsup_outputs) # For grad purposes
	for i, u in enumerate(unsup_outputs):
		temp[i] = F.softmax(u, dim=1)
	mean_prediction = temp.mean(0).unsqueeze(0) # Mean over the auxiliary outputs
	pw_loss = ((temp - mean_prediction)**2).mean(0) # Variance
	pw_loss = pw_loss.sum(1) # Sum over classes
	if size_average:
		return pw_loss.mean()
	return pw_loss.sum()

