import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/cropped_images/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='mmt', help='model_name')
parser.add_argument('--dataset', type=str,  default='la', help='dataset to use')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--iteration', type=int,  default=6000, help='GPU to use')
FLAGS = parser.parse_args()
#FLAGS.dataset
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model_" + FLAGS.dataset + "/"+FLAGS.model+"/"
test_save_path = "../model_acdc/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 4

with open(FLAGS.root_path + '/../test_acdc.list', 'r') as f:
    image_list = f.readlines()
# image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]
image_list = [FLAGS.root_path +item.replace('\n', '') for item in image_list]

def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()


    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 8), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    print('step one')
    metric = test_calculate_metric(FLAGS.iteration)
    print(metric)
    with open("../model_" + FLAGS.dataset + "/prediction.txt","a") as f:
        f.write(FLAGS.model + ": " + ", ".join(str(i) for i in metric) + "\n")
