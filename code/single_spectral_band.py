import numpy as np
import PIL.Image
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn import metrics

folder = ".././files-from-learn/data/"

multispectral_day01 = io.loadmat(folder + "multispectral_day01.mat")["immulti"]
annotation_day01 = PIL.Image.open(folder + "annotation_day01.png")
annotation_day01 = np.asarray(annotation_day01)

# RED: UNKNOWN, BLUE: MEAT, GREEN: FAT
unknown_channel_mask = (annotation_day01[:,:,0] > 0)
fat_channel_mask = (annotation_day01[:,:,1] > 0)
meat_channel_mask = (annotation_day01[:,:,2] > 0)

salami_mask = np.logical_or(
    np.logical_or(unknown_channel_mask, fat_channel_mask), 
    meat_channel_mask)

thresholds = []
confusion_matrices = []
error_rates = []
fat_masks = []
meat_masks = []

for i in range(19):
    channel = multispectral_day01[:,:,i]
    meat_mask_actual = np.logical_and(meat_channel_mask, salami_mask)
    masked_channel_meat = channel[meat_mask_actual]
    
    
    fat_mask_actual = np.logical_and(fat_channel_mask, salami_mask)
    masked_channel_fat = channel[fat_mask_actual]

    mean_meat = masked_channel_meat.mean()
    mean_fat = masked_channel_fat.mean()
    
    # Threshold values (gotten by solving normal_dist(mu_mean) = normal_dist(mu_fat))
    t = 1/2 * (mean_meat + mean_fat)
    thresholds.append(t)

    # If t >= mu_fat, then all values < t is classified as being fat, and all others meat
    if t >= mean_fat:
        fat_mask_predicted = np.logical_and(channel < t, salami_mask)
        meat_mask_predicted = np.logical_and(channel >= t, salami_mask)
        # meat_threshold_mask = np.logical_and(channel >= t, salami_mask)

        # Confusion matrix
        confusion_matrix = metrics.confusion_matrix(fat_mask_actual.astype(np.int32).flatten(), fat_mask_predicted.astype(np.int32).flatten())
        confusion_matrices.append(confusion_matrix)

        # Error rate 
        error = 1-(confusion_matrix[0,0] + confusion_matrix[1,1])/confusion_matrix.sum()
        error_rates.append(error)

    # Otherwise all values >= t is classified as being fat
    else:
        fat_mask_predicted = np.logical_and(channel >= t, salami_mask)
        meat_mask_predicted = np.logical_and(channel < t, salami_mask)
        # meat_threshold_mask = np.logical_and(channel < t, salami_mask)
        
        # Confusion matrix
        confusion_matrix = metrics.confusion_matrix(fat_mask_actual.astype(np.int32).flatten(), fat_mask_predicted.astype(np.int32).flatten())
        confusion_matrices.append(confusion_matrix)

        # Error rate 
        error = 1-(confusion_matrix[0,0] + confusion_matrix[1,1])/confusion_matrix.sum()
        error_rates.append(error)

    fat_masks.append(fat_mask_predicted)
    meat_masks.append(meat_mask_predicted)

    print(f"[{i+1}/19] --- mean_meat = {mean_meat} --- mean_fat = {mean_fat}")

print(thresholds)

# Confusion matrix format
# Row: True label, Column: Predicted label
#       False, True
# False   ?      ?
# True    ?      ?
print(confusion_matrices[19//2])

for i, e in enumerate(error_rates):
    print(i+1, e)

# Lowest error
best_spectral_band = np.argmin(error_rates)

print(f"Best spectral band: {1+best_spectral_band}")



plt.tight_layout()
plt.subplot(1,3,1)
plt.title("Fat mask")
fat_mask = fat_masks[best_spectral_band]
plt.imshow(fat_mask)

plt.subplot(1,3,2)
plt.title("Annotations")
plt.imshow(annotation_day01)

plt.subplot(1,3,3)
plt.title("Meat mask")
meat_mask = meat_masks[best_spectral_band]
plt.imshow(meat_mask)
plt.show()

plt.title("Meat prediction")
test = annotation_day01.copy()
test[:,:,0] = meat_mask * 255
plt.imshow(test)
plt.show()

# plt.matshow(annotation_day01)
# plt.show()