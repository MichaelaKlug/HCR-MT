import os
import sys
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
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
import matplotlib.pyplot as plt
from torch import nn

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


print(torch.cuda.is_available())
device = torch.device("cuda:0")

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
# from networks.hierarchical_vnet import VNet
from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
#from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

from dataloaders.acdc import acdc, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/cropped_images_binary', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='mmt', help='model_name')
parser.add_argument('--dataset', type=str,  default='la', help='dataset to use')

parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')

#trying 0.001 default was 0.01
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
    num_classes = 4
    #print('numb classes = ',num_classes)
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
        num_classes = 4
        patch_size = (112, 112, 8) 
        db_train = acdc(base_dir=train_data_path,
                           split='train',
                           transform = transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop(patch_size),
                              ToTensor(),
                              ]))
        db_test = acdc(base_dir=train_data_path,
                           split='test',
                           transform = transforms.Compose([
                               CenterCrop(patch_size),
                               ToTensor()
                           ]))
        labeled_idxs = list(range(16))
        unlabeled_idxs = list(range(16, 80))
    labs=list(range(0, 18))
    # unlabs=list(range(10, 19))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    #batch_sampler2 = TwoStreamBatchSampler(labs, [18], batch_size, batch_size-labeled_bs)
    def create_model(ema=False):
        # Network definition
        #, pyramid_has_dropout=True
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)

        #model = nn.DataParallel(net).cuda()
        model=net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    # def calculate_validation_loss(model, ema_model,val_loader):
    #     #print('here??')
    #     val_losses=[]
    #     model.eval()  # Set the model in evaluation mode
    #     ema_model.eval()
    #     with torch.no_grad():
    #         total_val_loss = 0.0
    #         num_batches = len(val_loader)
    #         #print(len(val_loader))
    #         for i_batch, val_batch in enumerate(val_loader):
    #             volume_batch, label_batch = val_batch['image'], val_batch['label']
    #             volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    #             noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
    #             ema_inputs = volume_batch + noise 

    #             # distill: 
    #             # student bs=4
    #             student_encoder_output,outputs= model(volume_batch)
                
    #             # teacher bs=2
    #             with torch.no_grad():
    #                 teacher_encoder_output,ema_output = ema_model(ema_inputs)
            
    #             loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            
    #             outputs_main_soft = F.softmax(outputs, dim=1)
    #             loss_seg_dice = losses.dice_loss(outputs_main_soft[:labeled_bs], label_batch[:labeled_bs])

    #             supervised_loss = 0.5*(loss_seg+loss_seg_dice)

    #             consistency_weight = get_current_consistency_weight(iter_num//150)
    #             consistency_dist = consistency_criterion(outputs, ema_output)
    #             consistency_dist = torch.mean(consistency_dist)
    #             consistency_loss = consistency_weight * consistency_dist
    #             loss = supervised_loss + consistency_loss #+ contrastive_loss #+contrastive_loss here
          
    #             total_val_loss+=loss.item()
    #             # val_losses.append(loss.item())

    #         # Calculate the average validation loss
    #         avg_val_loss = total_val_loss  / num_batches

    #     model.train()  # Set the model back to training mode
    #     ema_model.train()
    #     return avg_val_loss
    
    
    
    model = create_model() #student model
    ema_model = create_model(ema=True) #teacher model 

   
    loss_values=[]
    val_losses=[]


    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    # valloader= DataLoader(db_test, batch_sampler=batch_sampler2, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train() #student model
    ema_model.train() #teacher model
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter()
    logging.info("{} itertations per epoch".format(len(trainloader)))

    max_queue_size=100
    #create a queue to store all the negative keys --> maximum size is 100
    negative_keys=queue.Queue()

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        ave_loss=0.0
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
            student_encoder_output,outputs= model(volume_batch)
            
            # teacher bs=2
            with torch.no_grad():
                teacher_encoder_output,ema_output = ema_model(ema_inputs)
           
            ## calculate the loss
            # 1. L_sup bs=2 (labeled)
           
            #
            # print(outputs)
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_main_soft = F.softmax(outputs, dim=1)
            #loss_seg_dice = losses.dice_loss(outputs_main_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice = losses.dice_loss(outputs_main_soft[:labeled_bs], label_batch[:labeled_bs])

            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            ### hierarchical loss

            #contrastive loss
            
            # 2. L_con (labeled and unlabeled)
            ### hierarchical consistency
        
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(outputs, ema_output)
            consistency_dist = torch.mean(consistency_dist)
            consistency_loss = consistency_weight * consistency_dist

            # print('just before')
            # # losses.contrastive_loss(student_encoder_output,teacher_encoder_output,negative_keys)

            # # contrastive_loss=losses.contrastive_loss(student_encoder_output,teacher_encoder_output,negative_keys)
            # if negative_keys.qsize()>=100:
            #     negative_keys.get()
            # negative_keys.put(teacher_encoder_output)
            

            # total loss
            loss = supervised_loss + consistency_loss #+ contrastive_loss #+contrastive_loss here
            #print('type loss is ', type(loss.item()), " ", loss.item())
            #ave_loss+=loss.item()
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            writer.flush()

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            # val_losses.append(calculate_validation_loss(model, ema_model,valloader))
            if iter_num >= max_iterations:
                
                break
            time1 = time.time()
        
        
        if iter_num >= max_iterations:
            break
        
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    iterations = np.linspace(1, 6000, 6000, dtype=int)
    #loss_values = loss_values.cpu().detach().numpy()
    # plt.figure(figsize=(8, 6))
    # plt.plot(iterations, loss_values, marker='o', linestyle='-')
    # plt.title('Training Loss Over Iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # # Save the plot as an image file (e.g., PNG or PDF)
    # plt.savefig('training_loss_plot_32_val.png')

    # plt.figure(figsize=(8, 6))
    # plt.plot(iterations, val_losses, marker='o', linestyle='-')
    # plt.title('Validation Loss Over Iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.savefig('validation_loss_plot_32.png')
    # writer.flush()
    writer.close()
