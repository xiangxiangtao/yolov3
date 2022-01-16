import torch
import os

if __name__ == "__main__":
    weight_folder="/home/ecust/txx/project/yolov3_txx/weights"
    weight_name = "yolov3_gas_composite_2_epoch6.pth"  ###########################
    weight_path=os.path.join(weight_folder,weight_name)
    state_dict = torch.load(weight_path)
    weight_name_new="resaved.pth"
    torch.save(state_dict, os.path.join(weight_folder,weight_name_new), _use_new_zipfile_serialization=False)