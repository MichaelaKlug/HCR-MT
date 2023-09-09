import os
import sys
from tqdm import tqdm
import torch
#from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pdb
import os
import queue 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


print(torch.cuda.is_available())

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
# from networks.hierarchical_vnet import VNet
from networks.vnet_pyramid import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='mmt', help='model_name')
parser.add_argument('--dataset', type=str,  default='la', help='dataset to use')

parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')

parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### weight
parser.add_argument('--w0', type=float,  default=0.5, help='weight of p0')
parser.add_argument('--w1', type=float,  default=0.4, help='weight of p1')
parser.add_argument('--w2', type=float,  default=0.05, help='weight of p2')
parser.add_argument('--w3', type=float,  default=0.05, help='weight of p3')
### train
parser.add_argument('--mt', type=int,  default=0, help='mean teacher')
parser.add_argument('--mmt', type=int,  default=1, help='multi-scale mean teacher')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float,  default=1.0, help='temperature')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model_la/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
temperature = args.temperature
w0 = args.w0
w1 = args.w1
w2 = args.w2
w3 = args.w3
mt = args.mt
mmt = args.mmt

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'la':
        num_classes = 2
        patch_size = (112, 112, 80)
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform = transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop(patch_size),
                              ToTensor(),
                              ]))
        db_test = LAHeart(base_dir=train_data_path,
                           split='test',
                           transform = transforms.Compose([
                               CenterCrop(patch_size),
                               ToTensor()
                           ]))
        labeled_idxs = list(range(16))
        unlabeled_idxs = list(range(16, 80))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True, pyramid_has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model() #student model
    ema_model = create_model(ema=True) #teacher model 

    max_queue_size=100
    #create a queue to store all the negative keys --> maximum size is 100
    negative_keys=queue.Queue()


    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train() #student model
    ema_model.train() #teacher model
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    #writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()   

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise 

            # distill: 
            # student bs=4
            
            student_encoder_output,outputs_main, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)
            outputs_main_soft = F.softmax(outputs_main / temperature, dim=1)
            # outputs_aux1_soft = F.softmax(outputs_aux1 / temperature, dim=1)
            # outputs_aux2_soft = F.softmax(outputs_aux2 / temperature, dim=1)
            # outputs_aux3_soft = F.softmax(outputs_aux3 / temperature, dim=1)

            # teacher bs=2
            with torch.no_grad():
                teacher_encoder_output,ema_outputs_main, ema_outputs_aux1, ema_outputs_aux2, ema_outputs_aux3 = ema_model(ema_inputs)
            ema_outputs_main_soft = F.softmax(ema_outputs_main / temperature, dim=1)
            
            
            # ema_outputs_aux1_soft = F.softmax(ema_outputs_aux1 / temperature, dim=1)
            # ema_outputs_aux2_soft = F.softmax(ema_outputs_aux2 / temperature, dim=1)
            # ema_outputs_aux3_soft = F.softmax(ema_outputs_aux3 / temperature, dim=1)

            ## calculate the loss
            # 1. L_sup bs=2 (labeled)
            if mt: 
                ### the last layer
                loss_seg = F.cross_entropy(outputs_main[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice = losses.dice_loss(outputs_main_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            if mmt: 
                ##IS THIS WHERE I MUST CHANGE?
                ### hierarchical loss
                loss_seg_main = F.cross_entropy(outputs_main[:labeled_bs], label_batch[:labeled_bs])
                # loss_seg_aux1 = F.cross_entropy(outputs_aux1[:labeled_bs], label_batch[:labeled_bs])
                # loss_seg_aux2 = F.cross_entropy(outputs_aux2[:labeled_bs], label_batch[:labeled_bs])
                # loss_seg_aux3 = F.cross_entropy(outputs_aux3[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice_main = losses.dice_loss(outputs_main_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                # loss_seg_dice_aux1 = losses.dice_loss(outputs_aux1_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                # loss_seg_dice_aux2 = losses.dice_loss(outputs_aux2_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                # loss_seg_dice_aux3 = losses.dice_loss(outputs_aux3_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                loss_seg = loss_seg_main #+ w1 * loss_seg_aux1 + w2 * loss_seg_aux2 + w3 * loss_seg_aux3
                loss_seg_dice = loss_seg_dice_main #+ w1 * loss_seg_dice_aux1 + w2 * loss_seg_dice_aux2 + w3 * loss_seg_dice_aux3
                
                supervised_loss = 0.5*(loss_seg+loss_seg_dice)


                #contrastive loss
            cont_loss=losses.contrastive_loss(student_encoder_output,teacher_encoder_output,negative_keys)
            if negative_keys.qsize()>=100:
                negative_keys.get()
            negative_keys.put(teacher_encoder_output)
        


            # 2. L_con (labeled and unlabeled)
            ### hierarchical consistency
            ##IS THIS WHERE I MUST CHANGE?
            consistency_main_dist = (ema_outputs_main_soft - outputs_main_soft)**2
            # consistency_aux1_dist = (ema_outputs_aux1_soft - outputs_aux1_soft)**2
            # consistency_aux2_dist = (ema_outputs_aux2_soft - outputs_aux2_soft)**2
            # consistency_aux3_dist = (ema_outputs_aux3_soft - outputs_aux3_soft)**2
            consistency_main_dist = torch.mean(consistency_main_dist)
            # consistency_aux1_dist = torch.mean(consistency_aux1_dist)
            # consistency_aux2_dist = torch.mean(consistency_aux2_dist)
            # consistency_aux3_dist = torch.mean(consistency_aux3_dist)
            consistency_dist = consistency_main_dist #+ w1 * consistency_aux1_dist + w2 * consistency_aux2_dist + w3 * consistency_aux3_dist

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = consistency_weight * consistency_dist
            contrastive_loss= consistency_weight * cont_loss

            # total loss
            loss = supervised_loss + consistency_loss  + contrastive_loss#+contrastive_loss here

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            # writer.add_scalar('lr', lr_, iter_num)
            # writer.add_scalar('loss/loss', loss, iter_num)
            # writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            # writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            # writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            # writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            # writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                #writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_main_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                #writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                #writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                #writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_main_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                #writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                #writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    #writer.close()
