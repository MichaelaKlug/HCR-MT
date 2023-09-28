import torch
from torch.nn import functional as F
import numpy as np
import queue

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

#Loss formula from the paper 'Semi-supervise d me dical image segmentation via a triple d-uncertainty guided mean teacher model with contrastive learning'
#equation : -log( (exp(fs . ft+)) / (sum over all ft- of : exp(fs.ft-)) )
#where . is the inner product operation of vectors

#when adding to total loss- look at how other coefficients have been calculated for the other losses
# def contrastive_loss(fs,ft_plus,ftminus_queue):
#     print(type(fs))
#     print(type(fs[0]))
#     # fs=np.array(fs.cpu())
#     # ft_plus=np.array(ft_plus.cpu())
#     if ftminus_queue.qsize()==0:
#         #should this be return 1??
#         return 0
#     # fs_cpu = [tensor.cpu() for tensor in fs]
#     # ft_plus_cpu = [tensor.cpu() for tensor in ft_plus]

#     # fs_numpy = [tensor.numpy() for tensor in fs_cpu]
#     # ft_plus_numpy = [tensor.numpy() for tensor in ft_plus_cpu]
#     # Now calculate the inner product on the CPU
#     #inner_prod = sum(torch.inner(tensor1, tensor2) for tensor1, tensor2 in zip(fs, ft_plus))
#     inner_prod=0
#     for tense1,tense2 in zip(fs,ft_plus):
#         inner_prod += torch.inner(tense1, tense2)
#     # fs=fs.cpu()
#     # ft_plus=ft_plus.cpu()
#     #inner_prod=np.inner(fs,ft_plus)
#     numerator=np.exp(inner_prod)
#     minus_sum=0
#     for ftminus in ftminus_queue.qsize():
#         inner=np.inner(fs,ftminus)
#         denom=np.exp(inner)
#         minus_sum+=denom
#     answer=-1*np.log(numerator/minus_sum)
#     return answer

def contrastive_loss(fs,ft_plus,ftminus_queue):
     # Normalize the embeddings
        student_embeddings = F.normalize(fs, dim=1)
        teacher_embeddings = F.normalize(ft_plus, dim=1)

        # Calculate similarity scores
        logits = torch.matmul(student_embeddings, torch.cat([teacher_embeddings,ftminus_queue]))
        #logits /= self.temperature

        # Labels: positive samples followed by negative samples from the queue
        labels = torch.arange(logits.size(0), device=logits.device)

        # Compute the contrastive loss
        loss = F.cross_entropy(logits, labels)

        # Update the queue with student embeddings
        #self.update_queue(student_embeddings)

        return loss
def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
