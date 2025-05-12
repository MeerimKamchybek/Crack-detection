from multiprocessing import cpu_count
import torch
import time
from PIL import Image, ImageOps
import os.path
import sys
from model import *
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

cwd = os.getcwd()
os.chdir("F:\\Meerim\\5_tools\\Tool2")
#folder_path = "C:\\Users\\Anwender\\Desktop\\Neuer Ordner"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netAE = AE().to(device)
netAE.load_state_dict(torch.load(
    f"netAE_reg_bs16_ls64_best.pth"))
netAE.to(device)
netAE.eval()

transform = transforms.Compose([
    transforms.Resize(DIM),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(0.2, 0.1)
])

DIM = 64
# coordinates = ((1747, 590), (1050, 1432), (1548, 870))  # WZ2_C11
# coordinates = ((971, 1279), (1148, 1544)) #WZ2_C13
coordinates = ((971, 1279), (1148, 1544), (680, 1079))  # WZ2_C13
size = 250
threshold = 0.45


def crop_deploy(path, coordinates, size):
    fullpath = path
    im = Image.open(fullpath)

    crops = []
    for coord in coordinates:
        left, top = coord
        imCrop = im.crop((left, top, left + size, top + size))
        crops.append(imCrop)

    return crops


def get_results(path):
    with torch.no_grad():
        image_name = os.path.basename(path)
        crops = crop_deploy(path, coordinates, size)
        is_positive = False
        max_outlier_score = 0
        for i, crop in enumerate(crops):
            im = transform(crop).unsqueeze(0)
            im = im.to(device)
            bs = im.size(0)
            rec_image = netAE(im)
            rec_diff = ((rec_image.reshape(bs, -1) - im.reshape(bs, -1)) ** 2)
            rec_score = rec_diff.mean(dim=1)

            outlier_score = rec_score
            max_outlier_score = max(max_outlier_score, float(outlier_score))

        if max_outlier_score > threshold:
            is_positive = True

        #print(f"Image: {image_name}\nAnomaly Score: {max_outlier_score:.4f}\nThreshold: {threshold}\nPrediction: {'Positive' if is_positive else 'Negative'}\n")

    if is_positive:
        result = 1
        with open(os.path.join(os.path.dirname(path), "positive_images.txt"), "a") as f:
            f.write(f"{image_name}\n")
    else:
        result = 0

    return result


def main(folder_path):
    positive_count = 0
    total_files = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                path = os.path.join(root, file)
                result = get_results(path)
                if result == 1:
                    positive_count += 1
                total_files += 1
    print(f"{positive_count} out of {total_files} images were positive.")


def display_patches(image_path):
    crops = crop_deploy(image_path, coordinates, size)
    transformed_crops = [transform(crop).squeeze().numpy() for crop in crops]

    n = len(crops)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    fig.suptitle('Patches: Original (Top) and Transformed (Bottom)')

    for idx, (crop, transformed_crop) in enumerate(zip(crops, transformed_crops)):
        axes[0, idx].imshow(crop)
        axes[0, idx].axis('off')
        axes[1, idx].imshow(transformed_crop, cmap='gray')
        axes[1, idx].axis('off')

    plt.show()


def get_mean_anomaly_score(image_paths, normalization_mean, normalization_std, transform_base):
    transform = transforms.Compose(
        transform_base + [transforms.Normalize(normalization_mean, normalization_std)])
    total_anomaly_score = 0
    for path in image_paths:
        max_outlier_score = 0
        crops = crop_deploy(path, coordinates, size)
        for crop in crops:
            im = transform(crop).unsqueeze(0)
            im = im.to(device)
            bs = im.size(0)
            rec_image = netAE(im)
            rec_diff = ((rec_image.reshape(bs, -1) - im.reshape(bs, -1)) ** 2)
            rec_score = rec_diff.mean(dim=1)
            outlier_score = rec_score
            max_outlier_score = max(max_outlier_score, float(outlier_score))
        total_anomaly_score += max_outlier_score
    return total_anomaly_score / len(image_paths)


def optimize_normalization_factor(folder_path, min_factor=0.1, max_factor=1.0, num_steps=10):
    image_paths_positives = []
    image_paths_negatives = []

    positive_folder_path = os.path.join(folder_path, "Positives")
    negative_folder_path = os.path.join(folder_path, "Negatives")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if root == positive_folder_path:
                image_paths_positives.append(os.path.join(root, file))
            elif root == negative_folder_path:
                image_paths_negatives.append(os.path.join(root, file))

    step_size = (max_factor - min_factor) / num_steps
    transform_base = [
        transforms.Resize(DIM),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]

    best_normalization_mean = None
    best_normalization_std = None
    max_relative_difference = 0

    for mean_step in range(num_steps + 1):
        normalization_mean = min_factor + mean_step * step_size
        for std_step in range(num_steps + 1):
            normalization_std = min_factor + std_step * step_size
            mean_anomaly_score_positives = get_mean_anomaly_score(
                image_paths_positives, normalization_mean, normalization_std, transform_base)
            mean_anomaly_score_negatives = get_mean_anomaly_score(
                image_paths_negatives, normalization_mean, normalization_std, transform_base)
            relative_difference = abs(mean_anomaly_score_positives - mean_anomaly_score_negatives) / min(
                mean_anomaly_score_positives, mean_anomaly_score_negatives)

            if relative_difference > max_relative_difference:
                max_relative_difference = relative_difference
                best_normalization_mean = normalization_mean
                best_normalization_std = normalization_std

    return best_normalization_mean, best_normalization_std


if __name__ == "__main__":
    folder_path = "F:\\Meerim\\5_tools\\Tool2\\2_3"
    best_normalization_mean, best_normalization_std = optimize_normalization_factor(
        folder_path)
    print(
        f"Best normalization mean: {best_normalization_mean:.4f}, Best normalization std: {best_normalization_std:.4f}")

    # Update the normalization parameters in the transform
    transform.transforms[-1] = transforms.Normalize(
        best_normalization_mean, best_normalization_std)

    main(folder_path)
