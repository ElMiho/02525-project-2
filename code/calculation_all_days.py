#%%
import numpy as np
import PIL.Image
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn import metrics

from all_spectral_bands import S, mean_fat_list, mean_meat_list, cov_inverse, model_day
from single_spectral_band import thresholds, best_spectral_band, mean_fats, mean_meats, salami_mask
best_threshold = thresholds[best_spectral_band]

folder = ".././files-from-learn/data/"


multispectrals = []
annotations = []

days = ["01", "06", "13", "20", "28"]
for i in days:
    multispectral_day = io.loadmat(folder + f"multispectral_day{i}.mat")["immulti"]
    multispectrals.append(multispectral_day)
    
    annotation_day = PIL.Image.open(folder + f"annotation_day{i}.png")
    annotation_day = np.asarray(annotation_day)
    annotations.append(annotation_day)

# single spectral band analysis
error_rates_single_spectral_band = []
for i in range(len(days)):
    #if days[i] == model_day: continue

    channel = multispectrals[i][:,:,best_spectral_band]
    fat_mask_actual = np.logical_and(annotations[i][:,:,1] > 0, salami_mask)
    if mean_fats[best_spectral_band] > mean_meats[best_spectral_band]:
        fat_mask_predicted = np.logical_and(channel > best_threshold, salami_mask)
    else:
        fat_mask_predicted = np.logical_and(channel < best_threshold, salami_mask)

    confusion_matrix = metrics.confusion_matrix(fat_mask_actual.astype(np.int32).flatten(), fat_mask_predicted.astype(np.int32).flatten())
    error = 1 - (confusion_matrix[0,0] + confusion_matrix[1,1])/confusion_matrix.sum()
    error_rates_single_spectral_band.append(error)



# all spectral bands
p1 = 0.3
p2 = 1 - p1
error_rates_all_spectral_bands = []
for i in range(len(days)):
    #if days[i] == model_day: continue
    
    fat_mask_predicted = np.zeros((514,514))
    im_meat_predicted = np.zeros((514,514))

    data = multispectrals[i]
    data = np.asarray(data)

    for i in range(514):
        for j in range(514):
            if not salami_mask[i,j]: continue
            x = data[i,j,:]
            # Calculate S1 and S2
            s1 = S(x,mean_fat_list, cov_inverse, p1)
            s2 = S(x,mean_meat_list, cov_inverse, p2)
            if s1 >= s2:
                fat_mask_predicted[i,j] = 1
            else:
                im_meat_predicted[i,j] = 1

    confusion_matrix = metrics.confusion_matrix(fat_mask_actual.astype(np.int32).flatten(), fat_mask_predicted.astype(np.int32).flatten())
    error = 1 - (confusion_matrix[0,0] + confusion_matrix[1,1])/confusion_matrix.sum()
    error_rates_all_spectral_bands.append(error)

print(f"Model day: {model_day}")
print(f"Errors single: {error_rates_single_spectral_band}")
print(f"Errors all: {error_rates_all_spectral_bands}")
#%% Save data as csv. file for latex
error_to_csv = np.transpose(error_rates_all_spectral_bands)
np.savetxt(f"ErrorRate_all_day: {model_day}.csv", error_to_csv, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
# %%
