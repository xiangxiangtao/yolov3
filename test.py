from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim



class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    # print("start evaluating...")
    model.eval()#设置为验证模式

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    print("len_dataset={}".format(len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # 评估第batch_i批
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # print("evaluating batch {}".format(batch_i))
        # Extract labels
        # targets:  (batch_size, 6)，其中6指的是num, cls, center_x, center_y, widht, height，其中num指的第num个图片
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])#转换为左上右下形式
        targets[:, 2:] *= img_size#调整为原图大小

        # imgs = Variable(imgs.type(Tensor), requires_grad=False)#输入图片组成tensor
        imgs = imgs.type(Tensor)
        # print("imgs_size=",imgs.size())

        with torch.no_grad():
            _t['im_detect'].tic()
            outputs = model(imgs)#输入图片喂入model,得到outputs
            # print("outputs_size=", outputs.size())
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)#outputs进行NMS得到最终结果
            # print("nms_outputs=", outputs[0])
            detect_time = _t['im_detect'].toc(average=False)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)#评估一个batch样本的性能
        # print('*'*3,len(sample_metrics),'*'*3)

        # print('current_im_detect: {:d}/{:d} {:.3f}s'.format(batch_i + 1, len(dataset), detect_time))######################

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class#返回一个batch_size的评估指标


if __name__ == "__main__":
    # 2.解析输入的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")#1 when getting test time
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    # parser.add_argument("--weights_path", type=str, default="/content/drive/MyDrive/weights/YOLOv3_train_composite/yolov3_ckpt_5.pth", help="path to weights file")
    # parser.add_argument("--weights_path", type=str, default="/content/drive/MyDrive/weights/YOLOv3/yolov3_ckpt_5.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    opt = parser.parse_args()
    # 3.打印当前使用的参数
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.解析评估数据集的路径和class_names
    # data_config = parse_data_config(opt.data_config)
    test_path_composite = '/content/dataset/composite_gas_gmy_500_400/test/image'
    test_path_composite_1 = '/content/dataset/composite_gas_1_gmy_500_400/test/image'
    test_path_composite_2 = '/content/dataset/composite_gas_2_gmy_500_400/test/image'
    test_path_composite_18_1 = '/home/ecust/txx/project/yolov3_txx/data/dataset/composite/composite_18.1_gmy/test/image'

    test_path_real_annotated = '/content/dataset/real_annotated/test/image'
    test_path_real_annotated_1 = '/content/dataset/real_annotated_1/test/image'
    test_path_real_annotated_gmy = '/content/dataset/real_annotated_gmy/val/image'
    test_path_real_7_gmy = '/home/ecust/txx/project/yolov3_txx/data/dataset/real/real_7_gmy/val/image'
    
    test_path_list1=[test_path_composite,test_path_composite_1,test_path_composite_2,test_path_composite_18_1]
    test_path_list2=[test_path_real_annotated,test_path_real_annotated_1,test_path_real_annotated_gmy,test_path_real_7_gmy]

    # test_path = test_path_composite_18_1
    test_path=test_path_real_7_gmy##########################################
    print("test_path=",test_path)

    # class_names = load_classes(data_config["names"])
    if test_path in test_path_list1:
      class_names=['gas']
    elif test_path in test_path_list2:
      class_names=['smoke']
    print("class_names=",class_names)

    # 5.创建model
    # Initiate model
    model = Darknet(opt.model_def).to(device)

    # 6.加载模型的权重
    weight_folder="/home/ecust/txx/project/yolov3_txx/checkpoints/gas_yolov3_composite18.1"#######################
    print("weight_folder={}".format(weight_folder))
    weight_list=os.listdir(weight_folder)
    print("weight_list:",weight_list)
    for weight in weight_list:
        weight_num=int(weight[weight.index("ckpt")+5:weight.index(".pth")])
        if weight.endswith(".pth") and (weight_num not in []):
            print("-"*100)
            print("weight=",weight)
            weight_path=os.path.join(weight_folder,weight)
            model.load_state_dict(torch.load(weight_path))

            # 7.调用evaluate评估函数得到评估结果
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=test_path,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )

            # 8.打印每一种class的评估结果ap
            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

            # 9.打印平均的评估结果mAP
            print("mAP: {:.4f}".format(AP.mean()))
