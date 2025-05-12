#import of pytorch related libraries
import random
# Copying or moving files or folders can be automated using a Python module called shutil.
import shutil
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, random_split

#import of other libraries

import os.path, sys
from PIL import Image

#Cropping and putting to directories
path_folder_07_22_cam11 = "F:\\Tool_No_2\\20220706\\Cam11\\OK"
path_folder_08_22_cam11 = "F:\\Tool_No_2\\20220829\\Cam11\\OK"
path_folder_10_22_cam11 = "F:\\Tool_No_2\\20221010\\Cam11\\OK"
path_folder_12_22_cam11 = "F:\\Tool_No_2\\20221207\\Cam11\\OK"
path_folder_07_22_cam12 = "F:\\Tool_No_2\\20220706\\Cam12\\OK"
path_folder_08_22_cam12 = "F:\\Tool_No_2\\20220829\\Cam12\\OK"
path_folder_10_22_cam12 = "F:\\Tool_No_2\\20221010\\Cam12\\OK"
path_folder_12_22_cam12 = "F:\\Tool_No_2\\20221207\\Cam12\\OK"
path_folder_07_22_cam13 = "F:\\Tool_No_2\\20220706\\Cam13\\OK"
path_folder_08_22_cam13 = "F:\\Tool_No_2\\20220829\\Cam13\\OK"
path_folder_10_22_cam13 = "F:\\Tool_No_2\\20221010\\Cam13\\OK"
path_folder_12_22_cam13 = "F:\\Tool_No_2\\20221207\\Cam13\\OK"


list_path_folders = [path_folder_07_22_cam11, path_folder_08_22_cam11, path_folder_10_22_cam11, path_folder_12_22_cam11,
                     path_folder_07_22_cam12, path_folder_08_22_cam12, path_folder_10_22_cam12, path_folder_12_22_cam12,
                     path_folder_07_22_cam13, path_folder_08_22_cam13, path_folder_10_22_cam13, path_folder_12_22_cam13]

new_folder_path = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Evaluation_set\\Negatives"
for path_folder in list_path_folders:
    dirs = os.listdir(path_folder)
    print(len(dirs))
    # random.sample(population, k, *, counts=None). Random sampling without replacement.
    samples = random.sample(dirs, len(dirs)//70)
    for sample in samples:
        path = f"{path_folder}\\{sample}"
        new_path = f"{new_folder_path}\\{sample}"
        shutil.copyfile(path, new_path)
    print(f'copying of {len(samples)} images is completed')


#naming the samples accordingly
def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result
