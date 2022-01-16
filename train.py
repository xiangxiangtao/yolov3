from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")#409 batch
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/darknet53.conv.74",help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    best_map = 0.0

    # logger = Logger("/content/drive/MyDrive/logs/gas_composite_yolov3_1")#train composite
    # logger = Logger("/content/drive/MyDrive/logs/gas_composite_yolov3_2")#train real_annotated
    logger = Logger("/home/ecust/txx/project/yolov3_txx/logs/gas_yolov3")

    model_save_folder = r"/home/ecust/txx/project/yolov3_txx/checkpoints/gas_yolov3_current"
    os.makedirs(model_save_folder,exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)#在目标目录已存在的情况下不会触发FileExistsError异常

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    # composite
    train_path_composite = '/content/dataset/composite_gas_gmy_500_400/train/image'
    val_path_composite = '/content/dataset/composite_gas_gmy_500_400/val/image'
    test_path_composite = '/content/dataset/composite_gas_gmy_500_400/test/image'

    # composite1
    train_path_composite_1 = '/content/dataset/composite_gas_1_gmy_500_400/train/image'
    val_path_composite_1 = '/content/dataset/composite_gas_1_gmy_500_400/val/image'
    test_path_composite_1 = '/content/dataset/composite_gas_1_gmy_500_400/test/image'

    # composite2
    train_path_composite_2 = '/content/dataset/composite_gas_2_gmy_500_400/train/image'
    val_path_composite_2 = '/content/dataset/composite_gas_2_gmy_500_400/val/image'
    test_path_composite_2 = '/content/dataset/composite_gas_2_gmy_500_400/test/image'

    # composite_18.1
    # train_path_composite_18_1 = '/home/ecust/txx/project/yolov3_txx/data/dataset/composite/composite_18.1_gmy/train/image'
    trainval_path_composite_18_1 = '/home/ecust/txx/project/yolov3_txx/data/dataset/composite/composite_18.1_gmy/trainval/image'
    # val_path_composite_18_1 = '/home/ecust/txx/project/yolov3_txx/data/dataset/composite/composite_18.1_gmy/val/image'
    test_path_composite_18_1 = '/home/ecust/txx/project/yolov3_txx/data/dataset/composite/composite_18.1_gmy/test/image'

    # real_annotated
    train_path_real_annotated = '/content/dataset/real_annotated/train/image'
    val_path_gas_annotated = '/content/dataset/real_annotated/val/image'
    test_path_gas_annotated = '/content/dataset/real_annotated/test/image'

    # train_path=train_path_composite_18_1
    train_path=trainval_path_composite_18_1#####################################################################
    # val_path=val_path_composite_18_1########################################################################
    test_path=test_path_composite_18_1#####################################################################

    # change dateset(neu-det): change path,change class_names(config-coco.data-names),label(datasets.py-label_files),
    # change yolov3.cfg(filters,class),change log_dir
    # train_path = '/home/ecust/txx/dataset/object_detection_open_dataset/NEU-DET/train/image'
    # valid_path = '/home/ecust/txx/dataset/object_detection_open_dataset/NEU-DET/valid/image'
    class_names = load_classes(data_config["names"])#GasPlume

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)# 构建好模型后，对模型的参数初始化

    last_epoch=0

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
            last_epoch = int(opt.pretrained_weights[
                             opt.pretrained_weights.index('ckpt_') + 5:opt.pretrained_weights.index('.pth')]) - 1
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size,augment=True, multiscale=opt.multiscale_training)############
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,#在每个epoch开始的时候，对数据进行重新排序
        num_workers=opt.n_cpu,
        pin_memory=True,#如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # save train information
    save_folder = "/home/ecust/txx/project/yolov3_txx/train_info"
    os.makedirs(save_folder,exist_ok=True)
    save_path_txt = os.path.join(save_folder, "train_info_current.txt")
    with open(save_path_txt, "a") as f:
        f.write("-"*50+"\n")
        f.write(train_path+"\n")


    sum_loss=0
    for epoch in range(last_epoch+1,opt.epochs):
        print('*'*50)
        print("current epoch={}".format(epoch))
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            # imgs = Variable(imgs.to(device))
            # targets = Variable(targets.to(device), requires_grad=False)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad了。
            optimizer.zero_grad()

            loss, outputs = model(imgs, targets)
            # print(loss)
            loss.backward()

            optimizer.step()#梯度下降执行一步参数更新

            # if batches_done % opt.gradient_accumulations:
            #     # Accumulates gradient before each step
            #     optimizer.step()
            #     optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch+1, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # # Tensorboard logging
                # tensorboard_log = []
                # for j, yolo in enumerate(model.yolo_layers):
                #     for name, metric in yolo.metrics.items():
                #         if name != "grid_size":
                #             tensorboard_log += [(f"{name}_{j+1}", metric)]
                # tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            sum_loss+=loss.item()
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            # print(log_str)

            model.seen += imgs.size(0)

        skip_epoch = 1
        if (epoch+1)>skip_epoch and (epoch+1) % opt.evaluation_interval == 0:
            # Tensorboard logging
            tensorboard_log = []
            tensorboard_log += [("loss",sum_loss/len(dataloader))]
            logger.list_of_scalars_summary(tensorboard_log, epoch)
            sum_loss = 0

            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=test_path,####################################################################################
                iou_thres=0.5,####################################################################################
                conf_thres=0.3,#0.5
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print("---- best mAP={}".format(best_map))
            print(f"---- mAP {AP.mean()}")

        # if epoch % opt.checkpoint_interval == 0:
        #     torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % (epoch+1))
            print("*"*20)
            if True:
            # if (AP.mean())>best_map:#取最好的map保存
            #     print("saving best model...")
                model_path = os.path.join(model_save_folder,"yolov3_ckpt_{}.pth".format(epoch+1))
                torch.save(model.state_dict(), model_path)
                best_map = AP.mean()
                # print("best model saved")

            # save train information
            with open(save_path_txt,"a") as f:
                f.write("epoch{:0>3}: mAP={}/best_mAP={}\n".format(epoch+1,AP.mean(),best_map))