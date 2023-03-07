import numpy as np
import PIL.Image
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn import metrics

folder = ".././files-from-learn/data/"

multispectral_day01 = io.loadmat(folder + "multispectral_day01.mat")["immulti"]
annotation_day01 = PIL.Image.open(folder + "annotation_day01.png")
annotation_day01 = np.asarray(annotation_day01)

red_channel_mask = (annotation_day01[:,:,0] > 0)
green_channel_mask = (annotation_day01[:,:,1] > 0)
blue_channel_mask = (annotation_day01[:,:,2] > 0)

mean_meat_list = []
mean_fat_list = []

salami_mask = np.logical_or(
    np.logical_or(red_channel_mask, green_channel_mask), 
    blue_channel_mask)

for i in range(19):
    channel = multispectral_day01[:,:,i]
    masked_channel_meat = channel[red_channel_mask]
    # Assuming that both green and blue is fat 
    fat_mask_actual = np.logical_and(
        np.logical_or(green_channel_mask, blue_channel_mask), 
        salami_mask)
    masked_channel_fat = channel[fat_mask_actual]

    
    mean_meat = masked_channel_meat.mean()
    mean_fat = masked_channel_fat.mean()

    mean_fat_list.append(mean_fat)
    mean_meat_list.append(mean_meat)

# Calculate S1 (fat) and S2 (meat)
p1 = 0.3
p2 = 1 - p1