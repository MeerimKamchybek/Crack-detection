# dataloader can be used to iterate through the data, manage batches, transform the data, etc.
import pickle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import random
from model import *
from torch.utils.data import DataLoader
# optimization library
from torch import optim
# autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions
from torch import autograd
# nn is used as the basic building block for network graphs
from torch import nn
# saves a given Tensor into an image file
from torchvision.utils import save_image
# gets dataset and useful image transformations
from torchvision import datasets, transforms
# sampler classes are used to specify the sequence of indices/keys used in data loading
from torch.utils.data import sampler
# ML model evaluation metrics
from sklearn import metrics
import torch
import numpy as np
# model runtime
import time
# os module provides dozens of functions for interacting with the operating system (path etc.)
import os
import sys
sys.path.append(os.getcwd())
# importing AE architecture
# import network architecture
sys.path.append(os.getcwd())
# converting python object into bytes and vice versa

# Setting the variables
DIM = 64
BATCH_SIZE = 16
OUTPUT_DIM = 1 * DIM * DIM
LATENT_SIZE = 16
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

# function that transforms the data by adding Gaussian Noise


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# defining transformer for data augmentation


def one_class_dataloader(nw=0, bs_test=16, mode='test'):
    if mode == 'train':
        transform = transforms.Compose([
            # technique that encourages models to be less sensitive to changes in lighting and photo camera settings
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(
                0.8, 1.2), saturation=(0.9, 1.1)),
            # prevents overfitting to certain angles and edge directions
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(DIM),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # do not forget to adapt (standard deviation, mean) for your data
            # normalization helps to run the model faster
            transforms.Normalize(0.16, 0.52)
        ])
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.Resize(DIM),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # do not forget to adapt (standard deviation, mean) for your data
            transforms.Normalize(0.07,0.29)
        ])
    # setting the directory of the data folder
    path_train = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Train_patches"
    #path_test = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\Test_patches"
    #path_test = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\experimentation\\Eval_cam1"
    #path_test = "C:\\Users\\Precision\\Desktop\\2023_Tool2\Cam11\\Patches"
    path_test = "C:\\Users\\Precision\\Desktop\\2023_Tool2\\Cam13\\Patches\\Test_set"
    path_val = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\experimentation\\Eval_cam1"
    #path_eval = "F:\\Meerim\\5_tools\\Final_data\\multiple_patches\\Tool2\\experimentation\\Eval_cam1"
    #path_eval = "C:\\Users\\Precision\\Desktop\\2023_Tool2\\Cam11\\Evaluation_set"
    path_eval = "C:\\Users\\Precision\\Desktop\\2023_Tool2\\Cam13\\Patches\\Evaluation_set"

    # upload images from the folders
    image_dataset_train = datasets.ImageFolder(path_train, transform)
    image_dataset_test = datasets.ImageFolder(path_test, transform)
    image_dataset_val = datasets.ImageFolder(path_val, transform)
    image_dataset_eval = datasets.ImageFolder(path_eval, transform)
    # Create dataloaders for iterative access to images (full data set)
    trainloader = torch.utils.data.DataLoader(
        image_dataset_train, num_workers=nw, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        image_dataset_test, num_workers=nw, batch_size=bs_test, shuffle=True)
    valloader = torch.utils.data.DataLoader(
        image_dataset_val, num_workers=nw, batch_size=BATCH_SIZE, shuffle=False)
    evalloader = torch.utils.data.DataLoader(
        image_dataset_eval, num_workers=nw, batch_size=1, shuffle=False)
    return trainloader, testloader, valloader, evalloader


code_word = 'reg_bs16_ls64'


def train_autoencoder(code_word):
    # Denoising AE settings implemented. Noise and Gaussian blur are applied on input and the mean squared error is calculated with regard to the input without noise.
    # This is done to fight against overfitting and to increase the variability of the data and learn necessary invariances.
    corrupt_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 1.5))
    corrupt_noise = AddGaussianNoise(0, 0.2)
    # Feeding the function one_class_dataloader with training and validation samples
    dataloader, _, valloader, _ = one_class_dataloader(2, BATCH_SIZE)
    # Move model (memory and operations) that imported from AE functions to the CPU/GPU device
    autoencoder = AE().to(device)
    # Defining a Loss Function and Optimizer
    optimizer = optim.Adam(autoencoder.parameters(), 1 * 1e-4, (0.9, 0.99))
    crit = nn.MSELoss()
    # Create some empty arrays to store logs
    loss_epoch_list = []
    val_loss_list = []
    val_loss_min = 1e5
    # We loop over the training dataset multiple times (each time is called an epoch)
    for e in range(10):
        losses = []
        autoencoder.train()
        time_each = time.time()
        i = 0
        # We iterate through trainloader iterator
        # Each cycle is a minibatch
        for (x, _) in dataloader:
            # Move our data to CPU/GPU
            x = x.to(device)
            # Adding noise to input
            corrupted_x = corrupt_noise(x)
            # Forward Propagation
            rec_image = autoencoder(corrupted_x)
            # mean squared error is calculated with regard to the input without noise
            loss = crit(rec_image, x)
            if i % 50 == 0:
                print(f'loss of the batch {i}:{loss}')
            i += 1
            # Clear the gradients before training by setting to zero
            # Required for a fresh start, otherwise, the gradient would be a combination of the old gradient,
            # which you have already used to update your model parameters, and the newly-computed gradient.
            optimizer.zero_grad()
            # Back propagate to obtain the new gradients for all nodes
            loss.backward()
            # Update the gradients/weights
            optimizer.step()
            losses.append(loss.item())
        # Print time needed for each epoch
        print(f'time for epoch:{time.time() - time_each}')
        # switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) (batch normalization etc.)
        autoencoder.eval()
        rec_image = autoencoder(x)
        # Concatenates the given sequence of seq tensors in the given dimension
        d_input = torch.cat((x, rec_image), dim=0)
        # saving tensor into image file
        save_image(d_input * 0.16 + 0.52,
                   f'F:\\Meerim\\5_tools\\Tool2\\Encoder_training_images\\rec{code_word}' + str(e) + '.bmp')
        # validation:
        ########
        losses_val = []
        i = 0
        for (x, _) in valloader:
            x = x.to(device)
            rec_image = autoencoder(x)
            loss = crit(rec_image, x)
            if i < 10:
                print(f'val_loss of the batch {i}: {loss}')
            i += 1
            # The item() method extracts the loss’s value as a Python float
            losses_val.append(loss.item())
        losses_val_mean = np.mean(losses_val)
        val_loss_list.append(losses_val_mean)
        # To prevent overfitting we monitor the validation loss. val_loss_min is a threshold to whether quantify a
        # loss at some epoch as improvement or not.
        # If the difference of loss is below val_loss_min, it is quantified as no improvement
        if losses_val_mean < val_loss_min:
            val_loss_min = losses_val_mean
            torch.save(autoencoder.state_dict(
            ), f'F:\\Meerim\\5_tools\\Tool2\\models_to_save\\netAE_{code_word}_best.pth')
        #######
        torch.save(autoencoder.state_dict(
        ), f'F:\\Meerim\\5_tools\\Tool2\\models_to_save\\netAE_{code_word}_{e}.pth')
        loss_epoch_list.append(np.mean(losses))
        print(
            f'Epoch{e+1}, train loss: {np.mean(losses)}, val loss: {np.mean(losses_val)}')
        with open(f'F:\\Meerim\\5_tools\\Tool2\\pickles\\loss_epoch_list_{code_word}_bs{BATCH_SIZE}_latent{LATENT_SIZE}_dim{DIM}.pickle', 'wb') as handle:
            pickle.dump(loss_epoch_list, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'F:\\Meerim\\5_tools\\Tool2\\pickles\\val_loss_list_{code_word}_bs{BATCH_SIZE}_latent{LATENT_SIZE}_dim{DIM}.pickle', 'wb') as handle:
            pickle.dump(val_loss_list, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(mode='test'):
    model_number = 'reg_bs16_ls64_best'
    netAE = AE().to(device)
    # Loads a model’s parameter dictionary using a deserialized state_dict
    netAE.load_state_dict(torch.load(
        f"F:\\Meerim\\5_tools\\Tool2\\models_to_save\\netAE_{model_number}.pth"))
    # is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
    # For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you.
    # In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:
    netAE.eval()
    _, testloader, _, evalloader = one_class_dataloader(1, 1, mode='test')
    y_true, y_score, y_score_test, y_true_test, y_pred = [], [], [], [], []
    in_real, out_real, in_rec, out_rec = [], [], [], []
    fp_real, fp_fake, fn_real, fn_fake = [], [], [], []
    e = 0
    print('Evaluation is started')
    with torch.no_grad():
        path_save = f'F:\\Meerim\\5_tools\\Tool2\\Eval_images'
        for (x, label) in testloader:
            # freezing the size of 1st dimension
            bs = x.size(0)
            x = x.to(device)
            rec_image = netAE(x)
            # MSE reconstruction loss
            rec_diff = ((rec_image.reshape(bs, -1) - x.reshape(bs, -1))**2)
            rec_score = rec_diff.mean(dim=1)
            outlier_score = rec_score
            y_true.append(label)
            y_score.append(outlier_score.cpu())
        # y predicted
        y_score = np.concatenate(y_score)
        # y real
        y_true = np.concatenate(y_true)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        # optimal treshold calculation
        optimal_idx = np.argmax(tpr - 0.8*fpr)
        optimal_threshold = thresholds[optimal_idx]
        fpr_eval = fpr[optimal_idx]
        tpr_eval = tpr[optimal_idx]
        # using the optimal threshold calculated above for classification of positive and negative samples
        for (x, label) in testloader:
            bs = x.size(0)
            x = x.to(device)
            rec_image = netAE(x)
            y_true_test.append(label)
            idx_norm = (label == 0)
            in_real.append(x[idx_norm])
            in_rec.append(rec_image[idx_norm])
            idx_crack = (label != 0)
            out_real.append(x[idx_crack])
            out_rec.append(rec_image[idx_crack])
            # Loss function
            rec_diff = ((rec_image.reshape(bs, -1) - x.reshape(bs, -1))**2)
            rec_score = rec_diff.mean(dim=1)
            y_score_test.append(rec_score.cpu())
            if rec_score > optimal_threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
            # Saving images resulting in FP and FN
            if 1 in idx_norm:
                for i in range(rec_score[idx_norm].size()[0]):
                    if rec_score[idx_norm][i].item() > optimal_threshold:
                        fp_real.append(x[idx_norm][i])
                        fp_fake.append(rec_image[idx_norm][i])
                        save_image(
                            x[idx_norm][i], f'{path_save}\\fp\\fp_{e}_real.png', normalize=True)
                        save_image(
                            rec_image[idx_norm][i], f'{path_save}\\fp\\fp_{e}_fake.png', normalize=True)
            if 1 in idx_crack:
                for i in range(rec_score[idx_crack].size()[0]):
                    if rec_score[idx_crack][i].item() < optimal_threshold:
                        fn_real.append(x[idx_crack][i])
                        fn_fake.append(rec_image[idx_crack][i])
                        save_image(
                            x[idx_crack][i], f'{path_save}\\fn\\fn_{e}_real.png', normalize=True)
                        save_image(
                            rec_image[idx_crack][i], f'{path_save}\\fn\\fn_{e}_fake.png', normalize=True)
            e += 1
        y_score_test = np.concatenate(y_score_test)
        y_pred = np.asarray(y_pred)
        y_true_test = np.concatenate(y_true_test)
        # confusion metrics
        cm = metrics.confusion_matrix(y_true_test, y_pred)
        tp = cm[1, 1]
        fp = cm[0, 1]
        tn = cm[0, 0]
        fn = cm[1, 0]
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
    # torch.cat(in_real, dim=0): The torch.cat function concatenates tensors along a specified dimension, dim. In this case,
    # the dimension is 0, meaning that the tensors will be concatenated along the first dimension. The input to the function is a list of tensors in_real.
    # [:32]: This is a slicing operation that selects the first 32 elements of the result from the torch.cat function.
    in_real = torch.cat(in_real, dim=0)[:32]
    in_rec = torch.cat(in_rec, dim=0)[:32]
    out_real = torch.cat(out_real, dim=0)[:32]
    out_rec = torch.cat(out_rec, dim=0)[:32]
    save_image(torch.cat((in_real, in_rec), dim=0),
               f'{path_save}\\normal.bmp', normalize=True)
    save_image(torch.cat((out_real, out_rec), dim=0),
               f'{path_save}\\anomalies.bmp', normalize=True)
    print('auc:', metrics.roc_auc_score(y_true, y_score))
    print('Reasonable threshold:', optimal_threshold)
    # plotting the anomaly score distribution graphs with threshold bar to analyze the scale on which these distributions overlap and
    # trace changes in distribution shapes according to changes in hyperparameters
    plt.figure(figsize=(7, 4))
    graph_limit = 100
    plt.hist(y_score[y_true == 0], 100, density=True,
             alpha=0.5, color='blue', label='Normal')
    plt.hist(y_score[y_true == 1], 100, density=True,
             alpha=0.5, color='red', label='Anomalous')
    plt.vlines(optimal_threshold, 0, graph_limit,
               colors='k', linestyles='dashed', linewidth=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k")
               for c in ['blue', 'red']]
    labels = ['Normal', 'Anomalous']
    plt.legend(handles, labels)
    plt.xlabel('Anomaly score')
    plt.title("")
    ax = plt.gca()
    ax.set_ylim([0, graph_limit])
    # plt.show()
    plt.savefig(
        f"F:\\Meerim\\5_tools\\Tool2\\Eval_images\\graphs\\netAE_model{model_number}_bs{BATCH_SIZE}.png", dpi=1200)
    print(f'evaluation tpr {tpr_eval}, evaluation fpr: {fpr_eval}')
    print(f'true positive rate {tpr}, false positive rate: {fpr}')


device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else
#device = torch.device("cuda:0")

# train_autoencoder(code_word)
evaluate()
