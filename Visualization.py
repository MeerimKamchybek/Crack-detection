
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

#First experiment. Get reconstruction losses of plain autoencoder (11 epochs)
#Get the reconstruction loss of regularized autoencoder (15 epochs)
#Get the reconstruction loss of the MDN model (20 epochs) with or without x^2 term

#3 training graphs + 3 validation graphs (dotted)

#get files for graphs:
#extracting lists from saved files:
#plain_autoencoder:
#path_plain_train = "E:\\Andrei\\AE\\pickles\\FE\\loss_epoch_list_plain_bs16_latent128_dim64.pickle"
#train_loss_plain = pd.read_pickle(path_plain_train)
#path_plain_val = "E:\\Andrei\\AE\\pickles\\FE\\val_loss_list_plain_bs16_latent128_dim64.pickle"
#val_loss_plain = pd.read_pickle(path_plain_val)
#regularized_autoencoder:
path_reg_train = "F:\\Meerim\\5_tools\\Tool2\\pickles\\loss_epoch_list_reg_bs16_ls64_bs16_latent16_dim64.pickle"
train_loss_reg = pd.read_pickle(path_reg_train)
path_reg_val = "F:\\Meerim\\5_tools\\Tool2\\pickles\\val_loss_list_reg_bs16_ls64_bs16_latent16_dim64.pickle"
val_loss_reg = pd.read_pickle(path_reg_val)
#GMM-autoencoder:
#path_gmm_train ="E:\Andrei\AE+GMDN\pickles\The best\MSE_losses_list_bs16_latent128_ncomp8_dim64.pickle"
#train_loss_gmm = pd.read_pickle(path_gmm_train)
#path_gmm_val = "E:\Andrei\AE+GMDN\pickles\The best\MSE_losses_val_list_bs16_latent128_ncomp8_dim64.pickle"
#val_loss_gmm = pd.read_pickle(path_gmm_val)

plt.clf()


#val and train for reg
epochs = range(1,len(train_loss_reg)+1)
print(len(epochs))
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss_reg, alpha=0.9, marker='o', color='royalblue', label='Training')
plt.plot(epochs, val_loss_reg, alpha=0.9, marker='o', color='forestgreen', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Regularized Autoencoder. Training and Validation')
#To restrict:
#ax = plt.gca()
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_ylim([0, 1.05* max(val_loss_reg)])
plt.legend()
#plt.show()
plt.savefig(f"F:\\Meerim\\5_tools\\Results\\Tool2_plain_AE_train_val_loss.png", dpi=1200)
