import numpy as np
import PIL.Image
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn import metrics

def S(x, mean, cov_matrix_inv, p):
    res = np.dot(x, cov_matrix_inv @ mean) - 1/2 * np.dot(mean, cov_matrix_inv @ mean) + np.log(p)
    return res

# Same percentage of meat and fat
p1 = 0.5
p2 = 1 - p1

folder = ".././files-from-learn/data/"
model_day = "28"

multispectral_day01 = io.loadmat(folder + f"multispectral_day{model_day}.mat")["immulti"]
annotation_day01 = PIL.Image.open(folder + f"annotation_day{model_day}.png")
annotation_day01 = np.asarray(annotation_day01)

unknown_channel_mask = (annotation_day01[:,:,0] > 0)
fat_channel_mask = (annotation_day01[:,:,1] > 0)
meat_channel_mask = (annotation_day01[:,:,2] > 0)

mean_meat_list = []
mean_fat_list = []

salami_mask = np.logical_or(
    np.logical_or(unknown_channel_mask, fat_channel_mask), 
    meat_channel_mask)

# print(np.average(multispectral_day01, 2, np.repeat(np.expand_dims(meat_channel_mask.astype(np.int32), axis=2), 19, 2)).size())
data_vecs = []

fat_mask_actual = np.logical_and(fat_channel_mask, salami_mask)
meat_mask_actual = np.logical_and(meat_channel_mask, salami_mask)
for i in range(19):
    channel = multispectral_day01[:,:,i]
    masked_channel_meat = channel[meat_mask_actual]

    salami_channel = channel[salami_mask]
    data_vecs.append(salami_channel.flatten())

    masked_channel_fat = channel[fat_mask_actual]

    mean_meat = masked_channel_meat.mean()
    mean_fat = masked_channel_fat.mean()

    mean_fat_list.append(mean_fat)
    mean_meat_list.append(mean_meat)

mean_fat_list = np.asarray(mean_fat_list)
mean_meat_list = np.asarray(mean_meat_list)

data_vecs = np.asarray(data_vecs)
cov_matrix = np.cov(data_vecs)
cov_inverse = np.linalg.inv(cov_matrix)
# print(f"Covariance matrix, shape: {cov_matrix.shape}\n{cov_matrix}")

fat_mask_predicted = np.zeros((514,514))
im_meat_predicted = np.zeros((514,514))

data = multispectral_day01

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

if __name__ == "__main__":
    print(f"Error: {error}")

    plt.subplot(1,3,1)
    plt.title("Fat predicted")
    plt.imshow(fat_mask_predicted)

    plt.subplot(1,3,2)
    plt.title("Annotation")
    plt.imshow(annotation_day01)

    plt.subplot(1,3,3)
    plt.title("Meat predicted")
    plt.imshow(im_meat_predicted)

    plt.show()