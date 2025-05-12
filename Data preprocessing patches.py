#import of pytorch related libraries
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, random_split

#import of other libraries

import os.path, sys
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#dirs = os.listdir()
#mode = 'test'
mode = 'val'
#mode = 'train'
normality = 'positives'
#normality = 'negatives'
experiment = True

#Setting a path to get data
if mode == 'train':
        path = "F:\\Meerim\\5_tools\\Final_data\\raw\\Tool2\\Train_raw\\Negatives"
        path_save = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Train_patches\\Negatives"
elif mode == 'test':
        if normality == 'negatives':
            path = "F:\\Meerim\\5_tools\\Final_data\\raw\\Tool2\\Test_raw\\Negatives"
            path_save = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Test_patches\\Negatives"
        elif normality == 'positives':
            path = "F:\\Meerim\\5_tools\\Final_data\\raw\\Tool2\\Test_raw\\Positives"
            path_save = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Test_patches\\Positives"
elif mode == 'val':
    path = "F:\\Meerim\\5_tools\\Final_data\\raw\\Tool2\\Eval_raw\\Negatives"
    path_save = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Evaluation_set\\Negatives"
    #path = "F:\\Meerim\\Final_data\\raw\\Tool2\\Val_raw\\Negatives"
    #path_save = "F:\\Meerim\\Final_data\\Multiple_patches\\Tool2\\Val_patches\\Negatives"



if mode == 'test' or mode == 'val':
    coordinates = ((1214, 1257),(949,1432),(971,1279),(1148,1544),(273,339),(1673,626),(1548,870),(1747,590)) # Camera 1: Coordinates ((1548,870) (1747,590),) ((971,1279) (1148,1544)
    step_sizes = ((50, 100),(100,50),(50,100),(100,50), (50, 175), (10,10),(10,10),(10,10))
    num_steps = ((1, 2),(2,1),(1,2),(2,1),(1,2),(1,1),(1,1),(1,1))
elif mode == 'train':
    coordinates = ((1214, 1257),(949,1432),(971,1279),(1148,1544),(273,339),(1673,626),(1548,870),(1747,590)) #starting points for sliding windows tool 117
    #specify size of the step for each cropping area in the format (horizontal_steps, vertical steps)
    step_sizes = ((50, 100),(100,50),(50,100),(100,50), (50, 175), (10,10),(10,10),(10,10)) #step size (horizontal_steps, vertical steps) tool 117
    #step_sizes = ((80, 40), (80,30)) #for test negatives only
    num_steps = ((1, 2),(2,1),(1,2),(2,1),(1,2),(1,1),(1,1),(1,1)) #(horizontal_amount_of_crops, vertical_amount_of_crops)
    #num_steps = ((2,2), (2,2)) #for test negatives only
#if we need to mirror patch:
mirrored = False
#to specify the size of the patch
size = 250

def data_creation(path, path_save, coordinates, size, step_sizes, num_steps, mirrored = False):
    dirs = os.listdir(path)
    num = 0
    for item in dirs:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            _, prefix = item.split(".")
            name = item[:-4]
            for crop_num in range(len(coordinates)):
                camnumber = str(name[-2:])
                if ((camnumber == "11" and crop_num == 2) or (camnumber == "11" and crop_num == 3) or (
                        camnumber == "11" and crop_num == 4)):
                    continue
                if ((camnumber == "13" and crop_num == 0) or (camnumber == "13" and crop_num == 1) or (
                        camnumber == "13" and crop_num == 4) or (camnumber == "13" and crop_num == 5)
                        or (camnumber == "13" and crop_num == 6) or (camnumber == "13" and crop_num == 7)):
                    continue
                if ((camnumber == "12" and crop_num == 0) or (camnumber == "12" and crop_num == 1) or (
                        camnumber == "12" and crop_num == 2) or (camnumber == "12" and crop_num == 3)
                        or (camnumber == "12" and crop_num == 5) or (camnumber == "12" and crop_num == 6)
                        or (camnumber == "13" and crop_num == 7)):
                    continue
                left, top = coordinates[crop_num]
                step_w, step_h = step_sizes[crop_num]
                num_w, num_h = num_steps[crop_num]
                for i in range(num_w):
                    for j in range(num_h):
                        imCrop = im.crop((left + step_w*i, top + step_h*j, left + size + step_w*i, top + size + step_h*j))
                        if mirrored == True:
                            imCrop = ImageOps.mirror(imCrop)
                        imCrop.save(path_save + name + f'_{left}_{top}_{i}{j}.{prefix}', quality=100)
                        num += 1
    print(f'{num} images are created')

#to put cropped images into directories
data_creation(path, path_save, coordinates, size, step_sizes, num_steps, mirrored)





